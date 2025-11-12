import pandas as pd


def create_frequency_map(annotations_files: list[str]):
    """
    Create a frequency map of labels from annotation files.

    Args:
        annotations_files (list[str]): List of paths to annotation files.
    """
    frequency_map = {}
    for file_path in annotations_files:
        df = pd.read_parquet(file_path)
        