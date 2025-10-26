import os
import shutil
import kagglehub


# Declare download configs
download_configs = {
    "kaggle_dataset_path": 'MABe-mouse-behavior-detection',
    "save_path": "datasets"
}



# Log in Kaggle
kagglehub.login()

# Download the data from kaggle
mabe_mouse_behavior_detection_path = kagglehub.competition_download(download_configs['kaggle_dataset_path'])
print('Data source import complete.')

# Copy the downloaded data to destination folder
os.makedirs(download_configs['save_path'], exist_ok=True)
shutil.copytree(mabe_mouse_behavior_detection_path, download_configs['save_path'], dirs_exist_ok=True)