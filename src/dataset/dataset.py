import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .load_data import load_video_data



class VideoSegmentsDataset(Dataset):
    """
    Builds overlapping fixed-length windows (context_length) from videos.
    Caches the computed segments to disk for reuse.

    Parameters
    ----------
    lab_list : List[str]
        list of lab ids
    video_list : List[List[str]]
        list of lists of video ids for each lab (parallel to lab_list)
    metadata_path, tracking_folder, annotation_folder : str
        passed to load_video_data
    context_length : int
        number of frames per window
    overlap_frames : int
        overlap between consecutive windows (0 <= overlap_frames < context_length)
    skip_missing : bool
        if True, skip videos with missing files or metadata errors (otherwise raise)
    pickle_dir : str
        directory to store cached pickle file
    force_rebuild : bool
        if True, forces regeneration of segments even if cache exists
    """

    def __init__(
        self,
        lab_list,
        video_list,
        metadata_path,
        tracking_folder,
        annotation_folder,
        context_length,
        overlap_frames,
        skip_missing,
        pickle_dir,
        subset_name,
        force_rebuild=False
    ):
        assert len(lab_list) == len(video_list), "lab_list and video_list must be parallel lists"
        assert 0 <= overlap_frames < context_length, "overlap_frames must satisfy 0 <= overlap < context_length"

        self.lab_list = lab_list
        self.video_list = video_list
        self.metadata_path = metadata_path
        self.tracking_folder = tracking_folder
        self.annotation_folder = annotation_folder
        self.context_length = context_length
        self.overlap_frames = overlap_frames
        self.skip_missing = skip_missing
        self.pickle_dir = pickle_dir
        self.force_rebuild = force_rebuild

        os.makedirs(self.pickle_dir, exist_ok=True)
        self.pickle_path = os.path.join(self.pickle_dir, f"video_segments_{subset_name}.pkl")

        self.stride = context_length - overlap_frames
        self.segments = []

        # Try to load from cache
        if not self.force_rebuild and os.path.exists(self.pickle_path):
            try:
                with open(self.pickle_path, "rb") as f:
                    cache_data = pickle.load(f)
                self.segments = cache_data.get("segments", [])
                print(f"✅ Loaded {len(self.segments)} segments from cache: {self.pickle_path}")
                return
            except Exception as e:
                warnings.warn(f"Failed to load cached segments ({e}). Rebuilding...")

        # Otherwise build and save
        self._build_segments_index()
        self._save_segments_to_cache()

    # -------------------------
    # Parallelized segment computation
    # -------------------------
    @staticmethod
    def _compute_segments_for_video(args):
        lab_id, vid, tracking_folder, annotation_folder, context_length, stride, skip_missing = args
        tracking_path = os.path.join(tracking_folder, lab_id, f"{vid}.parquet")
        annotation_path = os.path.join(annotation_folder, lab_id, f"{vid}.parquet")

        if not os.path.exists(tracking_path) or not os.path.exists(annotation_path):
            msg = f"Skipping video {vid} in lab {lab_id}: "
            if not os.path.exists(tracking_path):
                msg += "tracking file missing. "
            if not os.path.exists(annotation_path):
                msg += "annotation file missing."
            if skip_missing:
                warnings.warn(msg)
                return []  # skip this video
            else:
                raise FileNotFoundError(msg)

        trk_df = pd.read_parquet(tracking_path)
        total_frames = trk_df["video_frame"].nunique()
        first_start = trk_df["video_frame"].min()
        last_valid_start = total_frames - context_length + 1

        starts = list(range(first_start, last_valid_start + 1, stride))
        if not starts or starts[-1] != last_valid_start:
            if starts and starts[-1] < last_valid_start:
                starts.append(last_valid_start)
            elif not starts:
                starts = [last_valid_start]

        return [(lab_id, vid, s, s + context_length - 1) for s in starts]


    def _build_segments_index(self):
        """Parallelized segment indexing with progress tracking."""
        all_tasks = [
            (lab_id, vid, self.tracking_folder, self.annotation_folder, self.context_length, self.stride, self.skip_missing)
            for lab_id, vids in zip(self.lab_list, self.video_list)
            for vid in vids
        ]

        max_workers = min(8, max(1, multiprocessing.cpu_count() // 2))
        segments = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(VideoSegmentsDataset._compute_segments_for_video, task)
                       for task in all_tasks]

            for f in tqdm(as_completed(futures), total=len(futures),
                          desc="Building segment index", unit="video"):
                try:
                    result = f.result()
                    segments.extend(result)
                except Exception as e:
                    if not self.skip_missing:
                        raise
                    warnings.warn(f"Skipping video due to error: {e}")

        self.segments = segments
        print(f"✅ Built {len(self.segments)} total segments.")

    def _save_segments_to_cache(self):
        """Save computed segments to disk using pickle."""
        try:
            with open(self.pickle_path, "wb") as f:
                pickle.dump({"segments": self.segments}, f)
            print(f"💾 Saved {len(self.segments)} segments to {self.pickle_path}")
        except Exception as e:
            warnings.warn(f"Failed to save segments cache: {e}")

    # -------------------------
    # Dataset interface
    # -------------------------
    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        lab_id, video_id, start_frame, stop_frame = self.segments[idx]

        (
            bodyparts_matrix,
            action_matrix,
            agent_matrix,
            target_matrix,
        ) = load_video_data(
            lab_id=lab_id,
            video_id=video_id,
            metadata_path=self.metadata_path,
            tracking_folder=self.tracking_folder,
            annotation_folder=self.annotation_folder,
            start_frame=start_frame,
            stop_frame=stop_frame,
        )

        return {
            "bodyparts": torch.from_numpy(np.asarray(bodyparts_matrix)).float(),
            "actions": torch.from_numpy(np.asarray(action_matrix)).float(),
            "agents": torch.from_numpy(np.asarray(agent_matrix)).float(),
            "targets": torch.from_numpy(np.asarray(target_matrix)).float(),
            "lab_id": lab_id,
            "video_id": video_id,
            "start_frame": start_frame,
            "stop_frame": stop_frame,
        }