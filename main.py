import os
from glob import glob
import argparse
import yaml
import json
import pandas as pd
import torch
from tqdm import tqdm
from src.dataset import filter_valid_data, split_multilabel_actions, get_dataloader
from src.model import MABeEncoder
from src.optimizer import get_adam_optimizer
from src.loss_function import BCELoss
from src.evals import MultiLabelEvaluator
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

    # Split dataset by force
    if dataset_configs.get('force_dataset_split', False) or not os.path.exists(dataset_configs['dataset_split_path']):

        # Filter only valid files having both tracking and annotation data
        valid_file_ids = filter_valid_data(
            tracking_folder_path=dataset_configs['train_tracking_folder_path'],
            annotation_folder_path=dataset_configs['train_annotation_folder_path']
        )
        print("Total valid files with both tracking and annotation data: ", len(valid_file_ids))

        # Create action and files map
        annotation_files = [os.path.join(dataset_configs['train_annotation_folder_path'], fid) for fid in valid_file_ids]
        print("Valid annotation files found: ", len(annotation_files))

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
        split_sets = split_multilabel_actions(action_files_list, seed=dataset_configs['split_seed'])
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

        # Saving the dataset splits
        print("Saving the dataset splits: ")
        split_dict = {
            'train': train_dict,
            'val': val_dict,
            'test': test_dict
        }
        os.makedirs(os.path.dirname(dataset_configs['dataset_split_path']), exist_ok=True)
        with open(dataset_configs['dataset_split_path'], 'w') as fp:
            json.dump(split_dict, fp, indent=4)
        print("=" * 50)
    else:
        with open(dataset_configs['dataset_split_path']) as fp:
            split_dict = json.load(fp)
        train_dict = split_dict['train']
        val_dict = split_dict['val']
        test_dict = split_dict['test']

    print("Labs found in each set: ")
    print("Train: ", len(train_dict.keys()))
    print("Val: ", len(val_dict.keys()))
    print("Test: ", len(test_dict.keys()))

    # Get dataloader for train, val and test
    train_loader = get_dataloader(
        lab_list=list(train_dict.keys()),
        video_list=list(train_dict.values()),
        metadata_path=dataset_configs['train_metadata_path'],
        tracking_folder=dataset_configs['train_tracking_folder_path'],
        annotation_folder=dataset_configs['train_annotation_folder_path'],
        context_length=train_configs['context_length'],
        batch_size=train_configs['batch_size'],
        overlap_frames=train_configs['overlap_frames'],
        skip_missing=train_configs['skip_missing'],
        pickle_dir=dataset_configs['dataset_index_dir'],
        subset_name='train',
        force_rebuild=dataset_configs['force_index_rebuild'],
        shuffle=True
    )
    val_loader = get_dataloader(
        lab_list=list(val_dict.keys()),
        video_list=list(val_dict.values()),
        metadata_path=dataset_configs['train_metadata_path'],
        tracking_folder=dataset_configs['train_tracking_folder_path'],
        annotation_folder=dataset_configs['train_annotation_folder_path'],
        context_length=train_configs['context_length'],
        batch_size=train_configs['batch_size'],
        overlap_frames=train_configs['overlap_frames'],
        skip_missing=train_configs['skip_missing'],
        pickle_dir=dataset_configs['dataset_index_dir'],
        subset_name='val',
        force_rebuild=dataset_configs['force_index_rebuild'],
        shuffle=False
    )
    # test_loader = get_dataloader(
    #     lab_list=list(test_dict.keys()),
    #     video_list=list(test_dict.values()),
    #     metadata_path=dataset_configs['train_metadata_path'],
    #     tracking_folder=dataset_configs['train_tracking_folder_path'],
    #     annotation_folder=dataset_configs['train_annotation_folder_path'],
    #     context_length=train_configs['context_length'],
    #     batch_size=train_configs['batch_size'],
    #     overlap_frames=train_configs['overlap_frames'],
    #     skip_missing=train_configs['skip_missing'],
    #     pickle_dir=dataset_configs['dataset_index_dir'],
    #     subset_name='test',
    #     force_rebuild=dataset_configs['force_index_rebuild'],
    #     shuffle=False
    # )
    
    # Get the model
    model = MABeEncoder(
        num_heads=train_configs['model']['num_heads']
    )

    # Get the evaluator
    evaluator = MultiLabelEvaluator()

    # Get the optimizer
    if train_configs['optimizer']['name'].lower().strip() == 'adam':
        optimizer = get_adam_optimizer(model, **train_configs['optimizer']['optimizer_params'])
    else:
        raise NotImplementedError("Selected optimizer support isnt added yet.")
    
    # Get the loss function
    if train_configs['loss_function']['name'].lower().strip() == 'bceloss':
        loss_function = BCELoss(**train_configs['loss_function']['loss_function_params'])

    # Start training
    train_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_function,
        evaluator=evaluator,
        n_epochs=2,
        checkpoint_path='dumps'
    )
    print("Training History: ")
    print(json.dumps(train_history, indent=4))
    
    return train_history


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