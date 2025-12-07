import joblib
import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
import numpy as np

# ============================== Pipeline Components
def process_port(p):
  return int(p, 16) if str(p).startswith('0x') else int(p)

def process_ports(ports: pd.DataFrame):
  return ports.map(process_port)

DEFAULT_RATE_SHIFT = 1.1
def shift_and_log(data, shift=DEFAULT_RATE_SHIFT): # 1.1 So that the output has no zero values, and a small change is not that significant
  return np.log10(data + shift)

COLUMNS_ORDERED = [
    'min', 'max', 'mean', 'stddev',
    'saddr', 'sport', 'daddr', 'dport',
    'srate', 'drate',
    'N_IN_Conn_P_SrcIP', 'N_IN_Conn_P_DstIP',
    'state_number', 'proto',
    'seq',
    'attack', 'category', 'subcategory'
]
srate_idx, drate_idx = [COLUMNS_ORDERED.index('srate'), COLUMNS_ORDERED.index('drate')]
class CombinedFeatureAdder(BaseEstimator, TransformerMixin):
  def __init__(self, normalize=True) -> None:
    super().__init__()
    self.normalize = normalize

  def fit(self, X, y=None):
    return self

  def transform(self, X: pd.DataFrame, y=None):
    srate_to_drate = np.log10(X.loc[:, 'srate'] + DEFAULT_RATE_SHIFT) / np.log10(X.loc[:, 'drate'] + DEFAULT_RATE_SHIFT)
    if self.normalize:
      return X.assign(srate_to_drate=np.log1p(srate_to_drate))
    else:
      return X.assign(srate_to_drate=srate_to_drate)

MODELS_PATH = os.environ.get('MODELS_PATH', './models/')
PREPROCESS_MODELS_PATH = os.environ.get('PREPROCESS_MODELS_PATH', f'./{MODELS_PATH}/preprocess')
ORIGINAL_MODELS_PATH = os.environ.get('ORIGINAL_MODELS_PATH', f'./{MODELS_PATH}/preprocess')

models = {
    # 'logisitc_regression_preprocess': {
    #     'name': 'Logistic Regression with Preprocessing',
    #     'path': f'{PREPROCESS_MODELS_PATH}/linear_regression.joblib',
    # },
    'random_forest_preprocess': {
        'name': 'Random Forest with Preprocessing',
        'path': f'{PREPROCESS_MODELS_PATH}/random_forest.joblib',
    },
}

for model in models:
    print(f'Loading model: {models[model]["name"]}')
    models[model]['model'] = joblib.load(models[model]['path'])


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

print(sample_df)
print(models['random_forest_preprocess']['model'].predict(sample_df))
