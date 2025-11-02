from pathlib import Path
import pandas as pd



def filter_valid_data(tracking_folder_path: str, annotation_folder_path: str):
    """
    Filters the dataset to include only valid data which has available annotations.

    Parameters
    ----------
    metadata_path : str
        Path to the metadata CSV file.
    tracking_folder_path : str
        Path to the folder containing tracking data files.
    annotation_folder_path : str
        Path to the folder containing annotation data files.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only valid data entries.
    """
    tracking_files = [str(p.relative_to(tracking_folder_path)) for p in Path(tracking_folder_path).rglob("*.parquet")]
    annotation_files = [str(p.relative_to(annotation_folder_path)) for p in Path(annotation_folder_path).rglob("*.parquet")]
    valid_files_set = set(tracking_files).intersection(set(annotation_files))
    return list(valid_files_set)