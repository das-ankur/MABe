import os
from glob import glob
import argparse
import yaml
import json
import pandas as pd
from tqdm import tqdm
from src.dataset import split_multilabel_actions, get_dataloader
from src.train import train_model



def _train(dataset_configs_path: str, train_configs_path: str):
    # Read dataset configs and training configs from YAML files
    with open(dataset_configs_path, 'r') as f:
        dataset_configs = yaml.safe_load(f)
    with open(train_configs_path, 'r') as f:
        train_configs = yaml.safe_load(f)
    print("=" * 50)
    print("Dataset Configs: ")
    if dataset_configs:
        print(json.dumps(dataset_configs, indent=4))
    else:
        print(dataset_configs)
    print("Train Configs: ")
    if train_configs:
        print(json.dumps(train_configs, indent=4))
    else:
        print(train_configs)
    print("=" * 50)
    
    # Create action and files map
    annotation_files = glob(os.path.join(dataset_configs['train_annotation_folder_path'], '**', '**.parquet'))
    print("Total annotation files found: ", len(annotation_files))

    # Create action file map from the annotation files
    action_files_list = {}
    for file_path in tqdm(annotation_files, desc="Processing annotation files", total=len(annotation_files)):
        df = pd.read_parquet(file_path)
        available_actions = df['action'].unique().tolist()
        for action in available_actions:
            if action in action_files_list:
                action_files_list[action].append(file_path)
            else:
                action_files_list[action] = [file_path]
    print("Found Actions: ", ', '.join(action_files_list.keys()))
    print('=' * 50)

    # Split the action files in different sets
    split_sets = split_multilabel_actions(action_files_list)
    print("Number of files in each set: ")
    for split_name, split_list in split_sets.items():
        print(split_name.title(), ": ", len(split_list))
    print("=" * 50)
    
    # Get lab and video list for all sets
    train_dict, val_dict, test_dict = {}, {}, {}
    for split_name, split_list in split_sets.items():
        for path in split_list:
            path_parts = path.split(os.sep)
            lab_id, video_id = path_parts[-2], os.path.splitext(path_parts[-1])[0]
            if split_name == 'train':
                if lab_id in train_dict:
                    train_dict[lab_id].append(video_id)
                else:
                    train_dict[lab_id] = [video_id]
            elif split_name == 'val':
                if lab_id in val_dict:
                    val_dict[lab_id].append(video_id)
                else:
                    val_dict[lab_id] = [video_id]
            elif split_name == 'test':
                if lab_id in test_dict:
                    test_dict[lab_id].append(video_id)
                else:
                    test_dict[lab_id] = [video_id]
    print("Labs found in each set: ")
    print("Train: ", len(train_dict.keys()))
    print("Val: ", len(val_dict.keys()))
    print("Test: ", len(test_dict.keys()))
    print("=" * 50)

    # Get dataloader for train, val and test: TODO


def main():
    print("Main function is initiated!")
    parser = argparse.ArgumentParser(description="Training script CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ---- Train Command ----
    train_parser = subparsers.add_parser("train", help="Run the training process")
    train_parser.add_argument(
        "--dataset-configs-path",
        type=str,
        default="configs/dataset_configs.yaml",
        help="Path to dataset configuration file (default: configs/dataset_configs.yaml)",
    )
    train_parser.add_argument(
        "--train-configs-path",
        type=str,
        default="configs/train_configs.yaml",
        help="Path to training configuration file (default: configs/train_configs.yaml)",
    )

    args = parser.parse_args()

    # ---- Command Execution ----
    if args.command == "train":
        print("Model train is initiated!")
        _train(args.dataset_configs_path, args.train_configs_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()