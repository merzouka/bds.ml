import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

import joblib
import os
import pandas as pd
from deploy.base.pipelines import *
from deploy.base.utils import *

MODELS_PATH = os.environ.get('MODELS_PATH', './models/')
PREPROCESS_MODELS_PATH = os.environ.get('PREPROCESS_MODELS_PATH', f'{MODELS_PATH}/yes-pre')
NO_PREPROCESS_MODELS_PATH = os.environ.get('NO_PREPROCESS_MODELS_PATH', f'{MODELS_PATH}/no-pre')


def load_models():
    """
    Load all models (with and without preprocessing).
    
    Returns:
        dict: Dictionary containing all loaded models with their metadata
    """
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
        # 'random_forest_with_preprocess': {
        #     'name': 'Random Forest with Preprocessing',
        #     'path': f'{PREPROCESS_MODELS_PATH}/random_forest.joblib',
        # },
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
    print("=" * 80)
    print("LOADING MODELS")
    print("=" * 80)
    print()
    
    # Load all models
    models = load_models()
    
    if models:
        # Test all models
        test_models(models)
    else:
        print("⚠ No models were loaded successfully. Please check the model paths.")
