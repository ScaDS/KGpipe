from pathlib import Path

from kgpipe.common import KG, DataFormat, Data

def get_test_data_path(file_name: str, check_exists: bool = True) -> Path:
    """
    get test data path
    """
    path = Path(__file__).parent / "test_data" / file_name
    if check_exists and not path.exists():
        raise FileNotFoundError(f"Test data file not found: {path}")
    return path

def get_test_kg(file_name: str, check_exists: bool = True) -> KG:
    """
    get test kg
    """
    path = get_test_data_path(file_name)
    return KG(file_name, name=file_name, path=path, format=DataFormat.RDF_TTL)

def get_data(file_name: str, format: DataFormat | None = None, check_exists: bool = True) -> Data:
    """
    get data from file
    """
    path = Path(__file__).parent / "test_data" / file_name
    if check_exists and not path.exists():
        raise FileNotFoundError(f"Test data file not found: {path}")

    # get Format from file extension
    if format is None:
        format = DataFormat.from_extension(path.suffix.lstrip("."))

    return Data(path=path, format=format)