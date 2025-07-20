import os

# This function creates a specific directory if it doesn't already exist
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)