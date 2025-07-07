import subprocess
from .config import log


def get_elf_data(file_path):
    elf_data = {k: 0 for k in [
        'comment', 'dynamic', 'dynstr', 'dynsym', 'eh_frame', 'eh_frame_hdr',
        'fini', 'fini_array', 'gnu.version', 'gnu.version_r', 'gnu.hash', 'got',
        'got.plt', 'init', 'init_array', 'interp', 'jcr', 'note.ABI-tag',
        'note.gnu.build-id', 'plt', 'rela.dyn', 'rela.plt', 'rodata', 'shstrtab',
        'strtab', 'symtab', 'text', 'data', 'bss', 'input_size']
    }

    try:
        out = subprocess.run(['readelf', '-S', file_path],
                             capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            for key in elf_data:
                if key in line:
                    parts = line.split()
                    if len(parts) > 4:
                        try:
                            elf_data[key] = int(parts[4], 16)
                        except ValueError:
                            pass
    except Exception as e:
        log.error(f"[ELF] readelf failed: {e}")

    try:
        out = subprocess.run(
            ['size', file_path], capture_output=True, text=True, check=True).stdout
        parts = out.splitlines()[1].split()
        elf_data.update({
            'text': int(parts[0]),
            'data': int(parts[1]),
            'bss': int(parts[2]),
            'input_size': int(parts[3])
        })
    except Exception as e:
        log.error(f"[ELF] size failed: {e}")

    return elf_data
