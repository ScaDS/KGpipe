from pathlib import Path

def get_test_data_path(relative_path: str):
    path =  Path(__file__).parent / "test_data" / relative_path
    path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"Test data path {path} does not exist")

    return path