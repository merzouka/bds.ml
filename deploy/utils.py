import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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
