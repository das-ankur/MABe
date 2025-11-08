import os
from cachetools import LRUCache
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
from src.scaling import compute_global_min_max, min_max_scaling
import tempfile
import uuid


# ---------- Global LRU Caches (size = 3) ----------
_METADATA_CACHE = LRUCache(maxsize=3)
_TRACKING_CACHE = LRUCache(maxsize=3)
_ANNOTATION_CACHE = LRUCache(maxsize=3)


# ---------- 6️⃣ Bodyparts, actions, initialization ----------
bodypart_list = [
    'body_center', 'ear_left', 'ear_right', 'forepaw_left',
    'forepaw_right', 'head', 'headpiece_bottombackleft', 'headpiece_bottombackright',
    'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 'headpiece_topbackleft',
    'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright',
    'hindpaw_left', 'hindpaw_right', 'hip_left', 'hip_right', 'lateral_left',
    'lateral_right', 'neck', 'nose', 'spine_1', 'spine_2', 'tail_base', 'tail_middle_1',
    'tail_middle_2', 'tail_midpoint', 'tail_tip'
]

action_dict = {
    'allogroom': 0, 'approach': 1, 'attack': 2, 'attemptmount': 3, 'avoid': 4,
    'biteobject': 5, 'chase': 6, 'chaseattack': 7, 'climb': 8, 'defend': 9,
    'dig': 10, 'disengage': 11, 'dominance': 12, 'dominancegroom': 13, 'dominancemount': 14,
    'ejaculate': 15, 'escape': 16, 'exploreobject': 17, 'flinch': 18, 'follow': 19,
    'freeze': 20, 'genitalgroom': 21, 'huddle': 22, 'intromit': 23, 'mount': 24,
    'rear': 25, 'reciprocalsniff': 26, 'rest': 27, 'run': 28, 'selfgroom': 29,
    'shepherd': 30, 'sniff': 31, 'sniffbody': 32, 'sniffface': 33,
    'sniffgenital': 34, 'submit': 35, 'tussle': 36
}



def load_video_data(
    lab_id: str,
    video_id: str,
    metadata_path: str,
    tracking_folder: str,
    annotation_folder: str,
    start_frame: int,
    stop_frame: int
):
    """
        Loads video tracking and annotation data for a specific lab and video, and returns 
        processed matrices for bodypart coordinates and actions.

        The function performs the following steps:
        1. Loads metadata, tracking, and annotation files (with caching).
        2. Filters metadata for the specified lab and video.
        3. Computes global min-max scaling values for bodypart coordinates.
        4. Constructs matrices for bodypart positions, actions, and agent/target annotations.
        5. Applies scaling to bodypart coordinates and populates action/agent/target matrices.

        Parameters
        ----------
        lab_id : str
            Identifier for the lab.
        video_id : str
            Identifier for the video.
        metadata_path : str
            Path to the CSV file containing metadata for all videos.
        tracking_folder : str
            Folder containing tracking parquet files for each lab/video.
        annotation_folder : str
            Folder containing annotation parquet files for each lab/video.
        start_frame : int
            First frame index to load (inclusive).
        stop_frame : int
            Last frame index to load (inclusive).

        Returns
        -------
        bodyparts_matrix : np.ndarray, shape (num_frames, num_mice * num_bodyparts * 2)
            Matrix containing scaled x and y coordinates of bodyparts for each mouse.
            Missing coordinates are filled with -1.0.
        action_matrix : np.ndarray, shape (num_frames, num_actions)
            Binary matrix indicating which actions occur in each frame.
        agent_matrix : np.ndarray, shape (num_frames, num_actions, num_mice)
            Binary matrix marking which mice are agents of each action per frame.
        target_matrix : np.ndarray, shape (num_frames, num_actions, num_mice)
            Binary matrix marking which mice are targets of each action per frame.

        Raises
        ------
        FileNotFoundError
            If metadata, tracking, or annotation files are missing.
        ValueError
            If no metadata row or multiple metadata rows exist for the given lab/video.

        Notes
        -----
        - Expects four mice with IDs [1, 2, 3, 4].
        - Bodypart list and action dictionary are fixed as defined in the function.
        - Uses global caches (_METADATA_CACHE, _TRACKING_CACHE, _ANNOTATION_CACHE, _SCALING_CACHE)
        to avoid reloading data for repeated calls.
        - Coordinates are min-max scaled using cached or computed global min-max values.
    """
    global _METADATA_CACHE, _TRACKING_CACHE, _ANNOTATION_CACHE, _SCALING_CACHE
    global bodypart, action_dict

    # ---------- 1️⃣ Load metadata ----------
    if metadata_path not in _METADATA_CACHE:
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        _METADATA_CACHE[metadata_path] = pd.read_csv(metadata_path)
    metadata_df = _METADATA_CACHE[metadata_path]

    # ---------- 2️⃣ Filter metadata ----------
    meta_row = metadata_df[
        (metadata_df["lab_id"] == lab_id) &
        (metadata_df["video_id"].astype(str) == str(video_id))
    ]
    if meta_row.empty:
        raise ValueError(f"No metadata row found for lab_id={lab_id}, video_id={video_id}")
    if len(meta_row) > 1:
        raise ValueError(f"Multiple metadata rows found for lab_id={lab_id}, video_id={video_id}")
    meta_row = meta_row.iloc[0]

    # ---------- 3️⃣ Build cache keys ----------
    trk_key = f"{lab_id}_{video_id}_tracking"
    ann_key = f"{lab_id}_{video_id}_annotation"

    annotation_path = os.path.join(annotation_folder, lab_id, f"{video_id}.parquet")
    tracking_path = os.path.join(tracking_folder, lab_id, f"{video_id}.parquet")

    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    if not os.path.exists(tracking_path):
        raise FileNotFoundError(f"Tracking file not found: {tracking_path}")

    # ---------- 4️⃣ Load tracking + annotation ----------
    if trk_key not in _TRACKING_CACHE:
        _TRACKING_CACHE[trk_key] = pd.read_parquet(tracking_path)
    trk_df = _TRACKING_CACHE[trk_key]

    if ann_key not in _ANNOTATION_CACHE:
        _ANNOTATION_CACHE[ann_key] = pd.read_parquet(annotation_path)
    ann_df = _ANNOTATION_CACHE[ann_key]

    # ---------- 5️⃣ Compute and cache global scaling values ----------
    scaling_info = compute_global_min_max(trk_df, tracking_path)
    x_min, x_max = scaling_info["x_min"], scaling_info["x_max"]
    y_min, y_max = scaling_info["y_min"], scaling_info["y_max"]

    expected_mice = [1, 2, 3, 4]
    num_bodyparts = len(bodypart_list)
    num_actions = len(action_dict)
    num_frames_requested = stop_frame - start_frame + 1

    # Create memory-mapped arrays in a cache directory
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate unique filenames for this operation
    unique_id = str(uuid.uuid4())
    bp_filename = os.path.join(cache_dir, f'bodyparts_{unique_id}.npy')
    act_filename = os.path.join(cache_dir, f'actions_{unique_id}.npy')
    agent_filename = os.path.join(cache_dir, f'agents_{unique_id}.npy')
    target_filename = os.path.join(cache_dir, f'targets_{unique_id}.npy')

    try:
        # Create memory-mapped arrays
        bodyparts_matrix = np.memmap(bp_filename, dtype=np.float32, mode='w+', 
                                   shape=(num_frames_requested, len(expected_mice) * num_bodyparts * 2))
        bodyparts_matrix.fill(-1.0)
        
        action_matrix = np.memmap(act_filename, dtype=np.float32, mode='w+', 
                                shape=(num_frames_requested, num_actions))
        action_matrix.fill(0)
        
        agent_matrix = np.memmap(agent_filename, dtype=np.float32, mode='w+', 
                               shape=(num_frames_requested, num_actions, len(expected_mice)))
        agent_matrix.fill(0)
        
        target_matrix = np.memmap(target_filename, dtype=np.float32, mode='w+', 
                                shape=(num_frames_requested, num_actions, len(expected_mice)))
        target_matrix.fill(0)
    except:
        # Fallback to regular numpy arrays if memory mapping fails
        print("⚠️ Memory mapping failed, falling back to regular numpy arrays")
        bodyparts_matrix = np.full((num_frames_requested, len(expected_mice) * num_bodyparts * 2), -1.0, dtype=np.float32)
        action_matrix = np.zeros((num_frames_requested, num_actions), dtype=np.float32)
        agent_matrix = np.zeros((num_frames_requested, num_actions, len(expected_mice)), dtype=np.float32)
        target_matrix = np.zeros((num_frames_requested, num_actions, len(expected_mice)), dtype=np.float32)

    # ---------- 7️⃣ Frame processing ----------
    frames_range = np.arange(start_frame, stop_frame + 1)
    trk_df_filtered = trk_df[(trk_df['video_frame'] >= start_frame) & (trk_df['video_frame'] <= stop_frame)].copy()
    ann_df_filtered = ann_df[(ann_df['start_frame'] <= stop_frame) & (ann_df['stop_frame'] >= start_frame)].copy()

    if 'bodypart' in trk_df_filtered.columns:
        trk_df_indexed = trk_df_filtered.set_index(['video_frame', 'mouse_id', 'bodypart'])
    else:
        trk_df_indexed = pd.DataFrame(columns=['x', 'y']).set_index(['video_frame', 'mouse_id', 'bodypart'])

    for i, vframe in enumerate(frames_range):
        if vframe in trk_df_indexed.index.get_level_values('video_frame'):
            vframe_trk_df = trk_df_indexed.loc[vframe]
        else:
            vframe_trk_df = pd.DataFrame()

        for mouse_idx, mice_id in enumerate(expected_mice):
            if not vframe_trk_df.empty and mice_id in vframe_trk_df.index.get_level_values('mouse_id'):
                mouse_trk_df = vframe_trk_df.loc[mice_id]
                x_cords = np.full(num_bodyparts, -1.0)
                y_cords = np.full(num_bodyparts, -1.0)

                for bpart_idx, bpart in enumerate(bodypart_list):
                    if bpart in mouse_trk_df.index:
                        record = mouse_trk_df.loc[bpart]
                        x_cords[bpart_idx] = record['x']
                        y_cords[bpart_idx] = record['y']

                # ✅ Use global scaling values
                x_cords_scaled = min_max_scaling(x_cords, x_min, x_max)
                y_cords_scaled = min_max_scaling(y_cords, y_min, y_max)

                start_idx = mouse_idx * num_bodyparts * 2
                bodyparts_matrix[i, start_idx:start_idx + num_bodyparts] = x_cords_scaled
                bodyparts_matrix[i, start_idx + num_bodyparts:start_idx + num_bodyparts * 2] = y_cords_scaled

        # ---------- 8️⃣ Annotation ----------
        vframe_ann_df = ann_df_filtered[
            (ann_df_filtered['start_frame'] <= vframe) & (ann_df_filtered['stop_frame'] >= vframe)
        ]

        if not vframe_ann_df.empty:
            for action in vframe_ann_df['action'].to_list():
                action_index = action_dict.get(action)
                if action_index is not None:
                    action_matrix[i, action_index] = 1.0

            for _, row in vframe_ann_df.iterrows():
                action = row['action']
                agent = row['agent_id']
                target = row['target_id']
                action_index = action_dict.get(action)
                if action_index is not None:
                    if agent in expected_mice:
                        agent_matrix[i, action_index, agent - 1] = 1.0
                    if target in expected_mice:
                        target_matrix[i, action_index, target - 1] = 1.0

     # ----------  Compute Attention Mask ----------
    # A frame is valid if it has at least one non -1.0 coordinate
    # bodyparts_matrix: (num_frames_requested, num_mice * num_bodyparts * 2)
    valid_frames = ~(bodyparts_matrix == -1.0).all(axis=1)
    attention_mask = valid_frames.astype(np.float32)  # 1.0 for valid, 0.0 for invalid

    # ---------- 🔟 Return and Cleanup ----------
    try:
        # Convert memmap arrays to numpy arrays for return
        return (
            np.array(bodyparts_matrix),
            attention_mask,
            np.array(action_matrix),
            np.array(agent_matrix),
            np.array(target_matrix)
        )
    finally:
        # Clean up memmap files
        try:
            os.unlink(bp_filename)
            os.unlink(act_filename)
            os.unlink(agent_filename)
            os.unlink(target_filename)
        except:
            pass  # Ignore cleanup errors