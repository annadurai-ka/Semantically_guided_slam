
def read_config(config_path):
    """
    Read configuration parameters from a text file.
    Expected format: each line contains 'key value' or 'key=value'.
    Returns a dictionary of parameters with numeric values parsed where possible.
    """
    params = {}
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.replace('=', ' ').split()
            if len(parts) != 2:
                continue
            key, val = parts
            try:
                # Try integer, then float, else leave as string
                params[key] = int(val) if '.' not in val else float(val)
            except ValueError:
                params[key] = val
    return params