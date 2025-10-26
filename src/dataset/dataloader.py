import os
from torch.utils.data import DataLoader
from .dataset import VideoSegmentsDataset




def get_dataloader(
    lab_list,
    video_list,
    metadata_path,
    tracking_folder,
    annotation_folder,
    context_length=16,
    batch_size=8,
    overlap_frames=4,
    num_workers=None,
    pin_memory=True,
    shuffle=True
):
    """
    Creates and returns a PyTorch DataLoader for the TrainVideoDataset.

    Args:
        lab_list (list): List of lab IDs.
        video_list (list): List of lists of video IDs corresponding to each lab.
        metadata_path (str): Path to metadata CSV or JSON file.
        tracking_folder (str): Path to folder containing tracking CSVs.
        annotation_folder (str): Path to folder containing annotations.
        context_length (int): Number of frames in each context window.
        batch_size (int): Batch size for training.
        overlap_frames (int): Number of overlapping frames between windows.
        num_workers (int, optional): Number of DataLoader workers.
            Defaults to half of available CPU cores.
        pin_memory (bool): Whether to pin memory for faster GPU transfer.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: PyTorch DataLoader wrapping TrainVideoDataset.
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
        overlap_frames=overlap_frames
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return dataloader