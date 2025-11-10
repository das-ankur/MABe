import os
from torch.utils.data import DataLoader
from .dataset import VideoSegmentsDataset



def get_dataloader(
    lab_list,
    video_list,
    metadata_path,
    tracking_folder,
    annotation_folder,
    context_length,
    batch_size,
    overlap_frames,
    skip_missing,
    pickle_dir,
    subset_name,
    n_splits=10,
    split_id=None,
    force_rebuild=False,
    balance_actions=True,
    num_workers=None,
    pin_memory=True,
    shuffle=True
):
    """
    Creates and returns a PyTorch DataLoader for the VideoSegmentsDataset.

    Args:
        lab_list (list): List of lab IDs.
        video_list (list): List of lists of video IDs corresponding to each lab.
        metadata_path (str): Path to metadata CSV or JSON file.
        tracking_folder (str): Path to folder containing tracking CSVs.
        annotation_folder (str): Path to folder containing annotations.
        context_length (int): Number of frames in each context window.
        batch_size (int): Batch size for training.
        overlap_frames (int): Number of overlapping frames between windows.
        skip_missing (bool): Whether to skip missing files/videos.
        pickle_dir (str): Directory to store cached segments and splits.
        subset_name (str): Name identifier for this dataset configuration.
        n_splits (int): Number of splits to create (default: 5).
        split_id (int or None): Which split to use (0 to n_splits-1). 
            If None, uses all data.
        force_rebuild (bool): Whether to rebuild cache even if it exists.
        balance_actions (bool): Whether to oversample to balance action labels (default: True).
        num_workers (int, optional): Number of DataLoader workers.
            Defaults to half of available CPU cores.
        pin_memory (bool): Whether to pin memory for faster GPU transfer.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: PyTorch DataLoader wrapping VideoSegmentsDataset.
        
    Examples:
        # Create dataset with all data
        >>> loader = get_dataloader(..., split_id=None)
        
        # Create dataset for specific split (e.g., for cross-validation)
        >>> train_loader = get_dataloader(..., split_id=0)
        >>> val_loader = get_dataloader(..., split_id=1)
        
        # Loop through all splits
        >>> for split_id in range(5):
        ...     loader = get_dataloader(..., split_id=split_id)
        ...     # train model with this split
    """

    # Automatically detect optimal num_workers if not provided
    if num_workers is None:
        try:
            cpu_cores = os.cpu_count() or 4
            num_workers = max(1, cpu_cores // 2)
        except Exception:
            num_workers = 4

    dataset = VideoSegmentsDataset(
        lab_list=lab_list,
        video_list=video_list,
        metadata_path=metadata_path,
        tracking_folder=tracking_folder,
        annotation_folder=annotation_folder,
        context_length=context_length,
        overlap_frames=overlap_frames,
        skip_missing=skip_missing,
        pickle_dir=pickle_dir, 
        subset_name=subset_name,
        n_splits=n_splits,
        split_id=split_id,
        force_rebuild=force_rebuild,
        balance_actions=balance_actions
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return dataloader


def get_all_split_dataloaders(
    lab_list,
    video_list,
    metadata_path,
    tracking_folder,
    annotation_folder,
    context_length,
    batch_size,
    overlap_frames,
    skip_missing,
    pickle_dir,
    subset_name,
    n_splits=5,
    force_rebuild=False,
    balance_actions=True,
    num_workers=None,
    pin_memory=True,
    shuffle=True
):
    """
    Convenience function to create dataloaders for all splits at once.
    
    Returns:
        list: List of DataLoaders, one for each split.
        
    Example:
        >>> dataloaders = get_all_split_dataloaders(..., n_splits=5)
        >>> for split_id, loader in enumerate(dataloaders):
        ...     print(f"Training on split {split_id}")
        ...     for batch in loader:
        ...         # training code
    """
    dataloaders = []
    
    for split_id in range(n_splits):
        loader = get_dataloader(
            lab_list=lab_list,
            video_list=video_list,
            metadata_path=metadata_path,
            tracking_folder=tracking_folder,
            annotation_folder=annotation_folder,
            context_length=context_length,
            batch_size=batch_size,
            overlap_frames=overlap_frames,
            skip_missing=skip_missing,
            pickle_dir=pickle_dir,
            subset_name=subset_name,
            n_splits=n_splits,
            split_id=split_id,
            force_rebuild=force_rebuild,
            balance_actions=balance_actions,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=shuffle
        )
        dataloaders.append(loader)
    
    return dataloaders