import sqlite3
import time
from .config import DB_FILE, log

conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()


def init_db():
    try:
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS processes (
            id INTEGER PRIMARY KEY,
            pid INTEGER,
            name TEXT,
            cpu_affinity TEXT,
            cpu_num INTEGER,
            cpu_percent FLOAT,
            cpu_times TEXT,
            nice INTEGER,
            start_time REAL,
            end_time REAL,
            tat REAL,
            memory_vms INTEGER,
            memory_data INTEGER,
            memory_rss INTEGER,
            memory_shared INTEGER,
            io_read_count INTEGER,
            io_write_count INTEGER,
            io_read_bytes INTEGER,
            io_write_bytes INTEGER,
            open_files INTEGER,
            connections INTEGER,
            threads INTEGER,
            status TEXT,
            env_vars INTEGER,
            bss INTEGER,
            data INTEGER,
            dynamic INTEGER,
            dynstr INTEGER,
            dynsym INTEGER,
            eh_frame INTEGER,
            comment INTEGER,
            interp INTEGER,
            eh_frame_hdr INTEGER,
            fini INTEGER,
            fini_array INTEGER,
            gnu_version INTEGER,
            gnu_version_r INTEGER,
            gnu_hash INTEGER,
            got INTEGER,
            init INTEGER,
            init_array INTEGER,
            plt INTEGER,
            got_plt INTEGER,
            rela_dyn INTEGER,
            rela_plt INTEGER,
            shstrtab INTEGER,
            jcr INTEGER,
            note_ABI_tag INTEGER,
            note_gnu_build_id INTEGER,
            rodata INTEGER,
            strtab INTEGER,
            symtab INTEGER,
            text INTEGER,
            input_size INTEGER,
            retrained INTEGER DEFAULT 0
        )
        ''')
        conn.commit()
    except Exception as e:
        log.error(f"[DB] Failed to initialize database: {e}")
        exit(1)

def insert_into_db(start_time, nice_val ,label,tat_predict,proc_info):
    end_time = time.time()
    tat = end_time-start_time
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO processes (
                pid, name, cpu_affinity, cpu_num, cpu_percent, cpu_times, nice, start_time, end_time, tat,
                memory_vms, memory_data, memory_rss, memory_shared,
                io_read_count, io_write_count, io_read_bytes, io_write_bytes,
                open_files, connections, threads,
                status, env_vars,
                bss, data, dynamic, dynstr, dynsym, eh_frame, comment, interp,
                eh_frame_hdr, fini, fini_array, gnu_version, gnu_version_r, gnu_hash,
                got, init, init_array, plt, got_plt, rela_dyn, rela_plt,
                shstrtab, jcr, note_ABI_tag, note_gnu_build_id, rodata,
                strtab, symtab, text, input_size
            ) VALUES (
                :pid, :name, :cpu_affinity, :cpu_num, :cpu_percent, :cpu_times, :nice, :start_time, :end_time, :tat,
                :memory_vms, :memory_data, :memory_rss, :memory_shared,
                :io_read_count, :io_write_count, :io_read_bytes, :io_write_bytes,
                :open_files, :connections, :threads,
                :status, :env_vars,
                :bss, :data, :dynamic, :dynstr, :dynsym, :eh_frame, :comment, :interp,
                :eh_frame_hdr, :fini, :fini_array, :gnu_version, :gnu_version_r, :gnu_hash,
                :got, :init, :init_array, :plt, :got_plt, :rela_dyn, :rela_plt,
                :shstrtab, :jcr, :note_ABI_tag, :note_gnu_build_id, :rodata,
                :strtab, :symtab, :text, :input_size
            )
        ''', {
            'pid': proc_info['pid'],
            'name': proc_info['name'] + " with_ml_new",
            'cpu_affinity': proc_info['cpu_affinity'],
            'cpu_num': proc_info['cpu_num'],
            'cpu_percent': proc_info['cpu_percent'],
            'cpu_times': proc_info['cpu_times'],
            'nice': nice_val,
            'start_time': start_time,
            'end_time': end_time,
            'tat': tat,
            'memory_vms': proc_info['memory_vms'],
            'memory_data': proc_info['memory_data'],
            'memory_rss': proc_info['memory_rss'],
            'memory_shared':proc_info['memory_shared'],
            'io_read_count': proc_info['io_read_count'],
            'io_write_count': proc_info['io_write_count'],
            'io_read_bytes': proc_info['io_read_bytes'],
            'io_write_bytes': proc_info['io_write_bytes'],
            'open_files': proc_info['open_files'],
            'connections': proc_info['connections'],
            'threads': proc_info['threads'],
            'status': proc_info['status'],
            'env_vars': proc_info['env_vars'],
            'bss': proc_info['bss'],
            'data': proc_info['data'],
            'dynamic': proc_info['dynamic'],
            'dynstr': proc_info['dynstr'],
            'dynsym': proc_info['dynsym'],
            'eh_frame': proc_info['eh_frame'],
            'comment': proc_info['comment'],
            'interp': proc_info['interp'],
            'eh_frame_hdr': proc_info['eh_frame_hdr'],
            'fini': proc_info['fini'],
            'fini_array': proc_info['fini_array'],
            'gnu_version': proc_info['gnu.version'],
            'gnu_version_r': proc_info['gnu.version_r'],
            'gnu_hash': proc_info['gnu.hash'],
            'got': proc_info['got'],
            'init': proc_info['init'],
            'init_array': proc_info['init_array'],
            'plt': proc_info['plt'],
            'got_plt': proc_info['got.plt'],
            'rela_dyn': proc_info['rela.dyn'],
            'rela_plt': proc_info['rela.plt'],
            'shstrtab': proc_info['shstrtab'],
            'jcr': proc_info['jcr'],
            'note_ABI_tag': proc_info['note.ABI-tag'],
            'note_gnu_build_id': proc_info['note.gnu.build-id'],
            'rodata': proc_info['rodata'],
            'strtab': proc_info['strtab'],
            'symtab': proc_info['symtab'],
            'text': proc_info['text'],
            'input_size': proc_info['input_size'],
        })
        conn.commit()
        print(f"[+] Stored process {proc_info['pid']} with TaT {tat:.2f}s in database.")
    except Exception as e:
        log.error(f"Failed to insert process data into database: {e}")
        print(
            f"Error: Failed to insert process data into database. Error Details: {e}")


def close_db():
    conn.close()
    log.info("[DB] Connection closed.")
