import os
from glob import glob
import pandas as pd

pd.set_option('display.max_columns', None)

dataset_configs = {
    'train_metadata_path': os.path.join('..', 'datasets', 'train.csv'),
    'train_tracking_dir_path': os.path.join('..', 'datasets', 'train_tracking'),
    'train_annotation_dir_path': os.path.join('..', 'datasets', 'train_annotation'),
}

tracking_files = glob(os.path.join(dataset_configs['train_tracking_dir_path'], '**', '*.parquet'), recursive=True)
annotation_files = glob(os.path.join(dataset_configs['train_annotation_dir_path'], '**', '*.parquet'), recursive=True)
print('Number of tracking files:', len(tracking_files))
print('Number of annotation files:', len(annotation_files))

tracking_file_ids = {os.path.splitext(os.path.basename(f))[0] for f in tracking_files}
annotation_file_ids = {os.path.splitext(os.path.basename(f))[0] for f in annotation_files}
valid_video_ids = sorted(tracking_file_ids.intersection(annotation_file_ids))
print('Number of valid video ids with both tracking and annotation files:', len(valid_video_ids))
print('Sample valid ids:', valid_video_ids[:10])

metadata_df = pd.read_csv(dataset_configs['train_metadata_path'])
print('Total rows in metadata:', len(metadata_df))
filtered = metadata_df[metadata_df['video_id'].isin(valid_video_ids)]
print('Rows matching valid_video_ids:', len(filtered))
print(filtered.head().to_dict(orient='records'))
