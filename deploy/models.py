import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

import joblib
import os
import pandas as pd
import gdown
from deploy.base.pipelines import *
from deploy.base.utils import *

GDRIVE_WITH_FOLDER_ID = "12R_m0PM9Tw-pqXlsjjD3ycXorKuM37Vf"
GDRIVE_WITHOUT_FOLDER_ID = "1kH3NvLa0-mXL1mTEqA41ayqmba_3VWNL"

MODELS_PATH = os.environ.get('MODELS_PATH', './models')
PREPROCESS_MODELS_PATH = os.environ.get('PREPROCESS_MODELS_PATH', f'{MODELS_PATH}/yes-pre')
NO_PREPROCESS_MODELS_PATH = os.environ.get('NO_PREPROCESS_MODELS_PATH', f'{MODELS_PATH}/no-pre')

models_config = {
    'knn_with_preprocess': {
        'name': 'KNN with Preprocessing',
        'path': f'{PREPROCESS_MODELS_PATH}/knn.joblib',
    },
    # 'knn_no_preprocess': {
    #     'name': 'KNN without Preprocessing',
    #     'path': f'{NO_PREPROCESS_MODELS_PATH}/knn.joblib',
    # },
    # 'xgboost_with_preprocess': {
    #     'name': 'XGBoost with Preprocessing',
    #     'path': f'{PREPROCESS_MODELS_PATH}/xgboost.joblib',
    # },
    # 'xgboost_no_preprocess': {
    #     'name': 'XGBoost without Preprocessing',
    #     'path': f'{NO_PREPROCESS_MODELS_PATH}/xgboost.joblib',
    # },
    'random_forest_with_preprocess': {
        'name': 'Random Forest with Preprocessing',
        'path': f'{PREPROCESS_MODELS_PATH}/random_forest.joblib',
    },
    'random_forest_no_preprocess': {
        'name': 'Random Forest without Preprocessing',
        'path': f'{NO_PREPROCESS_MODELS_PATH}/random_forest.joblib',
    },
    # 'gmm_with_preprocessin': {
    #     'name': 'GMM with Preprocessing',
    #     'path': f'{PREPROCESS_MODELS_PATH}/gmm.joblib',
    # },
    'gmm_no_preprocess': {
        'name': 'GMM without Preprocessing',
        'path': f'{NO_PREPROCESS_MODELS_PATH}/gmm.joblib',
    },
    # 'kmeans_with_preprocessin': {
    #     'name': 'K-means with Preprocessing',
    #     'path': f'{PREPROCESS_MODELS_PATH}/kmeans.joblib',
    # },
    'kmeans_no_preprocess': {
        'name': 'K-Means without Preprocessing',
        'path': f'{NO_PREPROCESS_MODELS_PATH}/kmeans.joblib',
    },
}

def download_folder_from_gdrive(folder_id, output_path):
    """
    Download entire folder from Google Drive.
    
    Args:
        folder_id (str): Google Drive folder ID
        output_path (str): Local path where folder should be saved
        
    Returns:
        bool: True if download was successful
    """
    try:
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        print(f"  Downloading folder to {output_path}...")
        os.makedirs(output_path, exist_ok=True)
        gdown.download_folder(
            url=url,
            output=output_path,
            quiet=False,
            use_cookies=False,
            remaining_ok=True
        )
        return True
    except Exception as e:
        print(f"  ✗ Error downloading folder: {str(e)}")
        return False


def ensure_models_downloaded():
    """
    Download all models from Google Drive if they don't exist locally.
    Uses folder download approach since individual file IDs need to be set up.
    """
    print("=" * 80)
    print("CHECKING FOR MISSING MODELS")
    print("=" * 80)
    print()
    
    # Create local directories
    os.makedirs(PREPROCESS_MODELS_PATH, exist_ok=True)
    os.makedirs(NO_PREPROCESS_MODELS_PATH, exist_ok=True)
    
    models_to_check = {
        'with': [os.path.basename(models_config[k]['path']) for k in models_config if 'with_pre' in k],
        'without': [os.path.basename(models_config[k]['path']) for k in models_config if 'no_pre' in k]
    }
    
    missing_with = []
    missing_without = []
    
    for model_file in models_to_check['with']:
        path = os.path.join(PREPROCESS_MODELS_PATH, model_file)
        if not os.path.exists(path):
            missing_with.append(model_file)
    
    for model_file in models_to_check['without']:
        path = os.path.join(NO_PREPROCESS_MODELS_PATH, model_file)
        if not os.path.exists(path):
            missing_without.append(model_file)
    
    if not missing_with and not missing_without:
        print("✓ All models are present locally. No download needed.\n")
        return True
    
    print(f"Missing models (with preprocessing): {missing_with if missing_with else 'None'}")
    print(f"Missing models (without preprocessing): {missing_without if missing_without else 'None'}")
    print()
    
    success = True
    
    if missing_with:
        print("Downloading models with preprocessing...")
        if not download_folder_from_gdrive(GDRIVE_WITH_FOLDER_ID, PREPROCESS_MODELS_PATH):
            success = False
    
    if missing_without:
        print("Downloading models without preprocessing...")
        print(f'Missing with: {missing_without}')
        # if not download_folder_from_gdrive(GDRIVE_WITHOUT_FOLDER_ID, NO_PREPROCESS_MODELS_PATH):
        #     success = False
    
    if success:
        print("\n✓ Successfully downloaded all missing models")
    else:
        print("\n⚠ Some models may not have been downloaded successfully")
    
    return success


def load_models():
    """
    Load all models (with and without preprocessing).
    Downloads from Google Drive if models don't exist locally.
    
    Returns:
        dict: Dictionary containing all loaded models with their metadata
    """
    
    # First, ensure all models are downloaded
    ensure_models_downloaded()
    
    print("=" * 80)
    print("LOADING MODELS")
    print("=" * 80)
    print()
    
    loaded_models = {}
    
    for model_key, model_info in models_config.items():
        try:
            print(f'Loading model: {model_info["name"]}')
            model_path = model_info['path']
            
            if not os.path.exists(model_path):
                print(f'  ⚠ Warning: Model file not found at {model_path}')
                continue
            
            # Load the model
            loaded_model = joblib.load(model_path)
            
            loaded_models[model_key] = {
                'name': model_info['name'],
                'path': model_path,
                'model': loaded_model,
            }
            print(f'  ✓ Successfully loaded {model_info["name"]}')
            
        except Exception as e:
            print(f'  ✗ Error loading {model_info["name"]}: {str(e)}')
    
    print(f'\nSuccessfully loaded {len(loaded_models)}/{len(models_config)} models\n')
    return loaded_models


def test_models(models):
    """
    Test all loaded models with sample data.
    
    Args:
        models (dict): Dictionary of loaded models
    """
    # Create sample dataframe
    sample_df = pd.DataFrame({
        "proto": ["tcp", "udp", "icmp"],
        "saddr": ["192.168.1.10", "10.0.0.5", "172.16.12.4"],
        "sport": [443, 53, 0],
        "daddr": ["172.217.3.110", "192.168.1.1", "8.8.8.8"],
        "dport": [51540, 60000, 0],
        "state_number": [3, 1, 0],
        "seq": [102934, 124, 1],
        "mean": [0.0042, 0.0003, 0.0],
        "stddev": [0.0021, 0.0001, 0.0],
        "min": [0.0021, 0.0001, 0.0],
        "max": [0.0063, 0.0005, 0.0],
        "srate": [120.5, 43.2, 0.1],
        "drate": [85.4, 12.8, 0.0],
        "N_IN_Conn_P_SrcIP": [12, 4, 1],
        "N_IN_Conn_P_DstIP": [150, 87, 10],
    })
    
    print("=" * 80)
    print("SAMPLE DATA:")
    print("=" * 80)
    print(sample_df)
    print("\n")
    
    print("=" * 80)
    print("MODEL PREDICTIONS:")
    print("=" * 80)
    
    for model_key, model_info in models.items():
        try:
            print(f"\n{model_info['name']}:")
            print("-" * 40)
            
            predictions = model_info['model'].predict(sample_df)
            print(f"Predictions: {predictions}")
            
            # Try to get category labels if available
            try:
                if hasattr(model_info['model'], 'category'):
                    categories = model_info['model'].category(predictions)
                    print(f"Categories: {categories}")
            except Exception as e:
                print(f"(Could not get category labels: {str(e)})")
                
        except Exception as e:
            print(f"\n{model_info['name']}:")
            print("-" * 40)
            print(f"✗ Error during prediction: {str(e)}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Load all models (will download from Google Drive if needed)
    models = load_models()
    
    if models:
        # Test all models
        test_models(models)
    else:
        print("⚠ No models were loaded successfully.")
        print("\nTroubleshooting:")
        print("1. Make sure the Google Drive folders are shared as 'Anyone with the link'")
        print("2. Update GDRIVE_WITH_FOLDER_ID and GDRIVE_WITHOUT_FOLDER_ID with correct IDs")
        print("3. Check your internet connection")
        print("4. Verify the folder structure: with/pipelines and without/pipelines")
        print(f"5. Check if models exist at: {MODELS_PATH}")
