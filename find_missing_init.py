import os

def find_missing_init(root_dir):
    missing = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if '__pycache__' in dirpath or '.git' in dirpath or '.mypy_cache' in dirpath or '.pytest_cache' in dirpath or '.venv' in dirpath:
            continue
        has_py = any(f.endswith('.py') for f in filenames)
        has_init = '__init__.py' in filenames
        if has_py and not has_init:
            missing.append(dirpath)
    return missing

if __name__ == '__main__':
    missing = find_missing_init('.')
    for d in missing:
        print(d)
