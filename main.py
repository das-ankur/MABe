import yaml
import pandas as pd
from src.dataset import get_dataloader

# Path to your YAML file
yaml_file = "dataset_configs.yaml"

with open(yaml_file, "r") as f:
    data = yaml.safe_load(f)

print(data)  # data is now a Python dictionary


# Get the list of labs and it's associated videos
temp_df = pd.read_csv(dataset_configs['train_metdata_path'])
lab_video_map = (
    temp_df.groupby("lab_id")["video_id"]
      .unique()
      .apply(list)
      .to_dict()
)
lab_ids = list(lab_video_map.keys())
video_ids = list(lab_video_map.values())
video_ids = [[int(v) for v in vids] for vids in video_ids]

dataloader = get_dataloader(
    lab_list=lab_ids,
    video_list=video_ids,
    metadata_path=dataset_configs['train_metdata_path'],
    tracking_folder=dataset_configs['train_tracking_folder'],
    annotation_folder=dataset_configs['train_annotation_folder'],
    context_length=100,
    batch_size=8
)

for i, batch in enumerate(dataloader):
    print(f"Batch {i}")
    print(batch.keys())  # should contain 'bodyparts', 'actions', 'agents', 'targets', etc.
    print(batch['bodyparts'].shape)
    if i == 2:  # just check first 3 batches
        break