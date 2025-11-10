import os
import json
import warnings
import pickle
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .load_data import load_video_data



class VideoSegmentsDataset(Dataset):
    """
    Builds overlapping fixed-length windows with balanced action sampling and n-fold splits.
    
    Features:
    - Oversamples underrepresented actions to match most frequent action
    - Creates n_splits for cross-validation or different training runs
    - Caches segments and splits to disk for reuse
    - Access data by split_id for reproducible experiments
    
    Parameters
    ----------
    lab_list : List[str]
        List of lab ids
    video_list : List[List[str]]
        List of lists of video ids for each lab (parallel to lab_list)
    metadata_path, tracking_folder, annotation_folder : str
        Paths to data files
    context_length : int
        Number of frames per window
    overlap_frames : int
        Overlap between consecutive windows (0 <= overlap_frames < context_length)
    skip_missing : bool
        If True, skip videos with missing files or metadata errors
    pickle_dir : str
        Directory to store cached pickle and JSON files
    subset_name : str
        Name identifier for this dataset configuration
    n_splits : int
        Number of splits to create (default: 5)
    split_id : int or None
        Which split to use (0 to n_splits-1). If None, uses all data.
    force_rebuild : bool
        If True, forces regeneration of segments and splits even if cache exists
    balance_actions : bool
        If True, oversample to balance action labels (default: True)
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
        n_splits=5,
        split_id=None,
        force_rebuild=False,
        balance_actions=True
    ):
        assert len(lab_list) == len(video_list), "lab_list and video_list must be parallel lists"
        assert 0 <= overlap_frames < context_length, "overlap_frames must satisfy 0 <= overlap < context_length"
        
        if split_id is not None:
            assert 0 <= split_id < n_splits, f"split_id must be between 0 and {n_splits-1}"

        self.lab_list = lab_list
        self.video_list = video_list
        self.metadata_path = metadata_path
        self.tracking_folder = tracking_folder
        self.annotation_folder = annotation_folder
        self.context_length = context_length
        self.overlap_frames = overlap_frames
        self.skip_missing = skip_missing
        self.pickle_dir = pickle_dir
        self.subset_name = subset_name
        self.n_splits = n_splits
        self.split_id = split_id
        self.force_rebuild = force_rebuild
        self.balance_actions = balance_actions

        os.makedirs(self.pickle_dir, exist_ok=True)
        
        # File paths for caching
        self.segments_pickle_path = os.path.join(
            self.pickle_dir, f"video_segments_{subset_name}.pkl"
        )
        self.splits_json_path = os.path.join(
            self.pickle_dir, f"splits_{subset_name}_n{n_splits}.json"
        )

        self.stride = context_length - overlap_frames
        self.all_segments = []
        self.segments = []  # Will hold segments for current split_id
        
        # Load or build segments
        self._load_or_build_segments()
        
        # Load or create splits
        self._load_or_create_splits()
        
        # Apply split filter if split_id is specified
        if self.split_id is not None:
            self._filter_by_split()
        else:
            self.segments = self.all_segments
            
        print(f"✅ Dataset ready with {len(self.segments)} segments" + 
              (f" (split {split_id}/{n_splits-1})" if split_id is not None else " (all data)"))

    def _load_or_build_segments(self):
        """Load segments from cache or build from scratch."""
        if not self.force_rebuild and os.path.exists(self.segments_pickle_path):
            try:
                with open(self.segments_pickle_path, "rb") as f:
                    cache_data = pickle.load(f)
                self.all_segments = cache_data.get("segments", [])
                print(f"✅ Loaded {len(self.all_segments)} segments from cache: {self.segments_pickle_path}")
                return
            except Exception as e:
                warnings.warn(f"Failed to load cached segments ({e}). Rebuilding...")
        
        # Build segments from scratch
        self._build_segments_index()
        self._save_segments_to_cache()

    def _build_segments_index(self):
        """Build segment index from video files."""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        
        all_tasks = [
            (lab_id, vid, self.tracking_folder, self.annotation_folder, 
             self.context_length, self.stride, self.skip_missing)
            for lab_id, vids in zip(self.lab_list, self.video_list)
            for vid in vids
        ]

        max_workers = min(4, max(1, multiprocessing.cpu_count() // 4))
        chunk_size = 10
        segments = []

        for i in range(0, len(all_tasks), chunk_size):
            chunk_tasks = all_tasks[i:i + chunk_size]
            
            try:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(self._compute_segments_for_video, task)
                        for task in chunk_tasks
                    ]

                    for f in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"Building segment index (chunk {i//chunk_size + 1}/{(len(all_tasks)+chunk_size-1)//chunk_size})",
                        unit="video"
                    ):
                        try:
                            result = f.result()
                            segments.extend(result)
                        except Exception as e:
                            if not self.skip_missing:
                                raise
                            warnings.warn(f"Skipping video due to error: {e}")
            
            except Exception as e:
                warnings.warn(f"Error processing chunk: {e}. Continuing with next chunk...")
                continue

        self.all_segments = segments
        print(f"✅ Built {len(self.all_segments)} total segments.")

    @staticmethod
    def _compute_segments_for_video(args):
        """Compute segments for a single video (static method for multiprocessing)."""
        lab_id, vid, tracking_folder, annotation_folder, context_length, stride, skip_missing = args
        tracking_path = os.path.abspath(os.path.join(tracking_folder, lab_id, f"{vid}.parquet"))
        annotation_path = os.path.abspath(os.path.join(annotation_folder, lab_id, f"{vid}.parquet"))

        if not os.path.exists(tracking_path) or not os.path.exists(annotation_path):
            msg = f"Skipping video {vid} in lab {lab_id}: "
            if not os.path.exists(tracking_path):
                msg += f"tracking file missing at {tracking_path}. "
            if not os.path.exists(annotation_path):
                msg += f"annotation file missing at {annotation_path}."
            if skip_missing:
                warnings.warn(msg)
                return []
            else:
                raise FileNotFoundError(msg)

        try:
            trk_df = pd.read_parquet(tracking_path, columns=["video_frame"])
            total_frames = trk_df["video_frame"].nunique()
            first_start = trk_df["video_frame"].min()
            last_valid_start = total_frames - context_length + 1
            
            del trk_df
            
            if last_valid_start < first_start:
                return []
                
            starts = range(first_start, last_valid_start + 1, stride)
            segments = []
            
            for start in starts:
                segments.append((lab_id, vid, start, start + context_length - 1))
                
            return segments
                
        except Exception as e:
            if skip_missing:
                warnings.warn(f"Error processing video {vid} in lab {lab_id}: {str(e)}")
                return []
            else:
                raise

    def _save_segments_to_cache(self):
        """Save computed segments to disk using pickle."""
        try:
            with open(self.segments_pickle_path, "wb") as f:
                pickle.dump({"segments": self.all_segments}, f)
            print(f"💾 Saved {len(self.all_segments)} segments to {self.segments_pickle_path}")
        except Exception as e:
            warnings.warn(f"Failed to save segments cache: {e}")

    def _load_or_create_splits(self):
        """Load splits from JSON or create new ones with balancing."""
        if not self.force_rebuild and os.path.exists(self.splits_json_path):
            try:
                with open(self.splits_json_path, "r") as f:
                    splits_data = json.load(f)
                self.splits = splits_data["splits"]
                self.action_stats = splits_data.get("action_stats", {})
                print(f"✅ Loaded {self.n_splits} splits from: {self.splits_json_path}")
                return
            except Exception as e:
                warnings.warn(f"Failed to load splits ({e}). Creating new splits...")
        
        # Create new splits with balancing
        self._create_balanced_splits()
        self._save_splits_to_json()

    def _create_balanced_splits(self):
        """Create balanced splits by oversampling underrepresented actions."""
        print("🔄 Creating balanced splits...")
        
        # Step 1: Load action labels for all segments
        print("  📊 Analyzing action distributions...")
        segment_actions = self._get_action_labels_for_segments()
        
        # Step 2: Group segments by actions
        action_to_segments = defaultdict(list)
        for idx, actions in enumerate(segment_actions):
            for action in actions:
                action_to_segments[action].append(idx)
        
        # Step 3: Find max count and balance
        if self.balance_actions and action_to_segments:
            max_count = max(len(indices) for indices in action_to_segments.values())
            print(f"  🎯 Target count per action: {max_count}")
            
            # Oversample each action to match max_count
            balanced_indices = []
            action_counts = {}
            
            for action, indices in action_to_segments.items():
                current_count = len(indices)
                action_counts[action] = current_count
                
                # Repeat indices to reach max_count
                repeats_needed = max_count // current_count
                remainder = max_count % current_count
                
                oversampled = indices * repeats_needed + indices[:remainder]
                balanced_indices.extend(oversampled)
            
            # Remove duplicates and shuffle
            balanced_indices = list(set(balanced_indices))
            np.random.shuffle(balanced_indices)
            
            self.action_stats = {
                "original_counts": action_counts,
                "target_count": max_count,
                "total_segments_before": len(self.all_segments),
                "total_segments_after": len(balanced_indices)
            }
            
            print(f"  ✨ Balanced dataset: {len(self.all_segments)} → {len(balanced_indices)} segments")
        else:
            balanced_indices = list(range(len(self.all_segments)))
            self.action_stats = {
                "total_segments": len(self.all_segments),
                "balancing": "disabled"
            }
        
        # Step 4: Create n_splits
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(balanced_indices)
        
        split_size = len(balanced_indices) // self.n_splits
        self.splits = []
        
        for i in range(self.n_splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size if i < self.n_splits - 1 else len(balanced_indices)
            split_indices = balanced_indices[start_idx:end_idx]
            self.splits.append(split_indices)
            print(f"  📦 Split {i}: {len(split_indices)} segments")

    def _get_action_labels_for_segments(self):
        """
        Extract action labels for all segments.
        Returns a list of sets, where each set contains action names present in that segment.
        """
        from .load_data import action_dict
        
        segment_actions = []
        
        for lab_id, video_id, start_frame, stop_frame in tqdm(
            self.all_segments, 
            desc="  Loading action labels",
            unit="segment"
        ):
            try:
                # Load only annotation data
                annotation_path = os.path.join(
                    self.annotation_folder, lab_id, f"{video_id}.parquet"
                )
                ann_df = pd.read_parquet(annotation_path)
                
                # Filter for this segment's frame range
                segment_ann = ann_df[
                    (ann_df['start_frame'] <= stop_frame) & 
                    (ann_df['stop_frame'] >= start_frame)
                ]
                
                # Get unique actions in this segment
                actions = set(segment_ann['action'].unique())
                segment_actions.append(actions)
                
            except Exception as e:
                warnings.warn(f"Error loading actions for {lab_id}/{video_id}: {e}")
                segment_actions.append(set())
        
        return segment_actions

    def _save_splits_to_json(self):
        """Save splits and statistics to JSON file."""
        try:
            splits_data = {
                "n_splits": self.n_splits,
                "splits": self.splits,
                "action_stats": self.action_stats,
                "config": {
                    "context_length": self.context_length,
                    "overlap_frames": self.overlap_frames,
                    "balance_actions": self.balance_actions,
                    "subset_name": self.subset_name
                }
            }
            
            with open(self.splits_json_path, "w") as f:
                json.dump(splits_data, f, indent=2)
            
            print(f"💾 Saved splits to: {self.splits_json_path}")
        except Exception as e:
            warnings.warn(f"Failed to save splits: {e}")

    def _filter_by_split(self):
        """Filter segments to only include those in the specified split."""
        split_indices = self.splits[self.split_id]
        self.segments = [self.all_segments[i] for i in split_indices]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        """Get a single data sample."""
        lab_id, video_id, start_frame, stop_frame = self.segments[idx]

        (
            bodyparts_matrix,
            attention_mask,
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
            "attention_mask": torch.from_numpy(np.asarray(attention_mask)).float(),
            "actions": torch.from_numpy(np.asarray(action_matrix)).float(),
            "agents": torch.from_numpy(np.asarray(agent_matrix)).float(),
            "targets": torch.from_numpy(np.asarray(target_matrix)).float(),
            "lab_id": lab_id,
            "video_id": video_id,
            "start_frame": start_frame,
            "stop_frame": stop_frame,
        }