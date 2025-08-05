import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from .config import HIGH_PRIORITY_LABEL, SECONDARY_PRIORITY_LABEL, log, LABELS_DICT, MODEL_TAT_PATH, SCALER_PATH, KMEANS_PATH
from .db import get_db_connection
import random
import json


class ModelManager:
    def __init__(self):
        self.model = joblib.load(MODEL_TAT_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.kmeans = joblib.load(KMEANS_PATH)
        self.high_priority = 0

    def prepare_features(self, proc_info):
        df = pd.DataFrame([proc_info]).drop(columns=['pid'])
        df_model = df[[
            'memory_vms', 'memory_data', 'io_write_count', 'bss', 'data', 'dynamic',
            'dynstr', 'eh_frame', 'eh_frame_hdr', 'fini', 'fini_array',
            'gnu.version', 'gnu.version_r', 'got', 'init', 'init_array', 'plt',
            'rela.dyn', 'rela.plt', 'shstrtab', 'strtab', 'text', 'input_size'
        ]].copy()
        df_model['vmsXmemorydata'] = df_model['memory_vms'] * \
            df_model['memory_data']
        df_model['squaredmemorydata'] = df_model['memory_data'] ** 2
        df_model['memorydata'] = df_model['memory_data'] ** 4
        df_model['memory_vmslog'] = np.sqrt(df_model['memory_vms'])
        df_model['memory_datasin'] = np.sin(df_model['memory_data']) ** 2
        df_model['textXmemory_data'] = np.log(
            df_model['memory_data'].replace(0, 1)) ** 2
        return df, df_model

    def predict_nice(self, proc_info, creation_time, pid_start_times):
        df, df_model = self.prepare_features(proc_info)
        scaled = self.scaler.transform(df_model)
        tat_pred = self.model.predict(scaled)
        df['tat_predicted'] = tat_pred
        df['io_write_count'] = df['io_write_count']*250
        df_kmeans = df[[
            'io_read_count', 'io_write_count', 'io_write_bytes', 'bss', 'data',
            'dynamic', 'dynstr', 'eh_frame', 'eh_frame_hdr', 'fini', 'fini_array',
            'gnu.version', 'gnu.version_r', 'got', 'init', 'init_array', 'plt',
            'rela.dyn', 'rela.plt', 'shstrtab', 'strtab', 'text', 'input_size',
            'tat_predicted'
        ]]
        label = self.kmeans.predict(df_kmeans)[0]
        if label == HIGH_PRIORITY_LABEL:
            if self.high_priority == 3:
                label = SECONDARY_PRIORITY_LABEL
            else:
                self.high_priority += 1

        nice_val = random.randint(*LABELS_DICT.get(label, (0, 0)))
        pid_start_times[proc_info['pid']] = (
            creation_time, proc_info, nice_val, label, tat_pred[0])

        # Signal the rebalancer that a new process was added
        # Import here to avoid circular dependency
        try:
            from .rebalance import signal_new_process
            signal_new_process()
        except ImportError:
            log.warning("[ML] Could not signal rebalancer due to import issue")

        log.debug(f"[ML] Predicted nice value: {nice_val}, label: {label}")
        return nice_val

    def incremental_retrain(self, extra_trees: int = 5, learning_rate: float = 0.05):
        """
        Incrementally retrain the XGBoost model with new data from the database.

        Args:
            extra_trees (int): Number of new trees to add.
            learning_rate (float): Learning rate for incremental training.
        """
        # Convert to debug - very frequent
        log.debug("[INCREMENTAL] Checking for new training data...")

        df_new = self._fetch_untrained_data()
        if df_new.empty:
            # Convert to debug - frequent when no data
            log.debug("[INCREMENTAL] No new data to retrain on.")
            return

        try:
            X_scaled, y = self._prepare_training_data(df_new)
            log.debug(
                f"[INCREMENTAL] New data prepared with {len(df_new)} samples.")
            booster = self.model.get_booster()
            updated_model = self._build_incremental_model(
                extra_trees, learning_rate)
            log.debug(
                f"[INCREMENTAL] Building updated model with {extra_trees} extra trees.")
            updated_model.fit(X_scaled, y, xgb_model=booster)
            log.debug("[INCREMENTAL] Model updated with new data.")
            # Save backup of the model
            joblib.dump(self.model, MODEL_TAT_PATH + ".bak")
            log.debug("[INCREMENTAL] Backup of the model saved.")
            self.model = updated_model
            joblib.dump(self.model, MODEL_TAT_PATH)

            self._mark_data_as_retrained()
            self._update_kmeans(df_new, X_scaled)
            log.debug("[INCREMENTAL] KMeans model updated with new data.")
            log.info(
                f"[INCREMENTAL] Model retrained with {len(df_new)} new samples.")

        except Exception as e:
            log.error(f"[INCREMENTAL] Retraining failed: {e}")

    def _fetch_untrained_data(self) -> pd.DataFrame:
        """Fetch new process data that hasn't been used for retraining."""
        conn, cursor = get_db_connection()
        query = "SELECT * FROM processes WHERE retrained = 0 AND tat IS NOT NULL"
        return pd.read_sql_query(query, conn)

    def _prepare_training_data(self, df: pd.DataFrame):
        """Prepare feature matrix and target vector from raw process data."""
        feature_cols = [
            'memory_vms', 'memory_data', 'io_write_count', 'bss', 'data', 'dynamic',
            'dynstr', 'eh_frame', 'eh_frame_hdr', 'fini', 'fini_array',
            'gnu.version', 'gnu.version_r', 'got', 'init', 'init_array', 'plt',
            'rela.dyn', 'rela.plt', 'shstrtab', 'strtab', 'text', 'input_size'
        ]
        log.debug(
            f"[INCREMENTAL] Preparing training data with columns: {df.columns.tolist()}")
        df = df.rename(columns={
            'gnu_version': 'gnu.version',
            'gnu_version_r': 'gnu.version_r',
            'rela_dyn': 'rela.dyn',
            'rela_plt': 'rela.plt'
        })
        log.debug(
            f"[INCREMENTAL] Renamed columns for consistency: {df.columns.tolist()}")
        X = df[feature_cols].copy()
        X['vmsXmemorydata'] = X['memory_vms'] * X['memory_data']
        X['squaredmemorydata'] = X['memory_data'] ** 2
        X['memorydata'] = X['memory_data'] ** 4
        X['memory_vmslog'] = np.sqrt(X['memory_vms'])
        X['memory_datasin'] = np.sin(X['memory_data']) ** 2
        X['textXmemory_data'] = np.log(X['memory_data'].replace(0, 1)) ** 2

        X_scaled = self.scaler.transform(X)
        y = df['tat']
        return X_scaled, y

    def _build_incremental_model(self, extra_trees: int, learning_rate: float):
        """Build a new XGBoost model with updated parameters."""
        config = json.loads(self.model.get_booster().save_config())
        # log.debug("[DEBUG] Booster config:\n%s", json.dumps(config, indent=4))

        num_trees = int(config["learner"]["gradient_booster"]
                        ["gbtree_model_param"]["num_trees"])
        total_trees = num_trees + extra_trees

        params = {
            "learning_rate": learning_rate,
            "max_depth": int(config["learner"]["gradient_booster"]["tree_train_param"]["max_depth"]),
            "min_child_weight": float(config["learner"]["gradient_booster"]["tree_train_param"]["min_child_weight"]),
            "gamma": float(config["learner"]["gradient_booster"]["tree_train_param"]["gamma"]),
            "subsample": float(config["learner"]["gradient_booster"]["tree_train_param"]["subsample"]),
            "colsample_bytree": float(config["learner"]["gradient_booster"]["tree_train_param"]["colsample_bytree"]),
            "reg_alpha": float(config["learner"]["gradient_booster"]["tree_train_param"].get("reg_alpha", 0)),
            "reg_lambda": float(config["learner"]["gradient_booster"]["tree_train_param"].get("reg_lambda", 1)),
            "n_estimators": total_trees,
            "objective": config["learner"]["learner_train_param"]["objective"],
            "tree_method": config["learner"]["gradient_booster"]["gbtree_train_param"]["tree_method"]
        }

        return XGBRegressor(**params)

    def _mark_data_as_retrained(self):
        """Mark all new samples in the database as retrained."""
        conn, cursor = get_db_connection()
        cursor.execute(
            "UPDATE processes SET retrained = 1 WHERE retrained = 0")
        conn.commit()

    def _update_kmeans(self, df_new: pd.DataFrame, X_scaled):
        """Update the KMeans model with new features."""
        # log cols name debug
        log.debug(
            f"[KMEANS] Updating KMeans with new data columns: {df_new.columns.tolist()}")

        df_new['tat_predicted'] = self.model.predict(X_scaled)
        # log debug for tat_predicted
        log.debug(
            f"[KMEANS] New data with tat_predicted: {df_new[['tat_predicted']].head()}")
        # Check KMeans features
        log.debug(f"[KMEANS] KMeans features: {self.kmeans.feature_names_in_}")
        kmeans_features = [
            'io_read_count', 'io_write_count', 'io_write_bytes', 'bss', 'data',
            'dynamic', 'dynstr', 'eh_frame', 'eh_frame_hdr', 'fini', 'fini_array',
            'gnu.version', 'gnu.version_r', 'got', 'init', 'init_array', 'plt',
            'rela.dyn', 'rela.plt', 'shstrtab', 'strtab', 'text', 'input_size',
            'tat_predicted'
        ]
        df_new = df_new.rename(columns={
            'gnu_version': 'gnu.version',
            'gnu_version_r': 'gnu.version_r',
            'rela_dyn': 'rela.dyn',
            'rela_plt': 'rela.plt'
        })
        df_kmeans = df_new[[
            col for col in kmeans_features if col in df_new.columns]]
        joblib.dump(self.kmeans, KMEANS_PATH + ".bak")
        log.debug("[KMEANS] Backup of KMeans model saved.")
        self.kmeans.partial_fit(df_kmeans)
        joblib.dump(self.kmeans, KMEANS_PATH)
        log.info("[KMEANS] KMeans model updated with new data.")
