import random
from collections import defaultdict



def split_multilabel_actions(actions_files_list, split_ratio=(0.7, 0.15, 0.15), seed=42):
    """
    Splits multilabel action data by files (ensuring no overlap across train/val/test).
    Keeps track of action coverage in each split.

    Parameters
    ----------
    actions_files_list : dict[str, list[str]]
        Mapping from action name to list of files containing that action.
    split_ratio : tuple
        Train, val, and test ratios.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        {
            'train': set(files),
            'val': set(files),
            'test': set(files)
        }
    dict
        action_coverage: {split_name: {action_name: count}}
    """
    random.seed(seed)

    # Step 1: Collect all unique files
    all_files = set()
    for files in actions_files_list.values():
        all_files.update(files)
    all_files = list(all_files)
    random.shuffle(all_files)

    # Step 2: Split globally by file
    n_total = len(all_files)
    n_train = int(split_ratio[0] * n_total)
    n_val = int(split_ratio[1] * n_total)

    train_files = set(all_files[:n_train])
    val_files = set(all_files[n_train:n_train + n_val])
    test_files = set(all_files[n_train + n_val:])

    splits = {'train': train_files, 'val': val_files, 'test': test_files}

    # Step 3: Compute action coverage per split
    action_coverage = {k: defaultdict(int) for k in splits}

    for action, files in actions_files_list.items():
        for split_name, split_files in splits.items():
            count = sum(f in split_files for f in files)
            if count > 0:
                action_coverage[split_name][action] = count

    return splits, action_coverage

