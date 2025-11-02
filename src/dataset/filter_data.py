import os
import glob
import pandas as pd



def filter_valid_data(metadata_path: str, tracking_folder_path: str, annotation_folder_path: str):
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
    meta_df = pd.read_csv(metadata_path)
    lab_ids = os.listdir(tracking_folder_path)
    valid_lab_ids = set(lab_ids).intersection(set(meta_df['lab_id'].unique().tolist()))
    valid_videos_per_lab = {}
    for lab_id in valid_lab_ids:
        tracking_files = glob.glob(os.path.join(tracking_folder_path, lab_id, '*.parquet'))
        annotation_files = glob.glob(os.path.join(annotation_folder_path, lab_id, '*.parquet'))
        valid_videos = set(
            os.path.splitext(os.path.basename(f))[0] for f in annotation_files
        ).intersection(
            set(os.path.splitext(os.path.basename(f))[0] for f in tracking_files)
        )
        valid_videos_per_lab[lab_id] = valid_videos
    return valid_videos_per_lab
