import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_RATE_SHIFT = 1.1

def process_categories(cat):
    """Process category names to standardize them"""
    if str(cat).lower().startswith('theft'):
        return 'Theft'
    if str(cat).lower().startswith('normal'):
        return 'Normal'
    if cat == 'DoS HTTP':
        return 'DoS TCP'
    if cat == 'DDoS HTTP':
        return 'DDoS TCP'
    return cat

def process_port(p):
    """Convert port to integer, handling hex format"""
    return int(p, 16) if str(p).startswith('0x') else int(p)

def process_ports(ports: pd.DataFrame):
    """Apply port processing to DataFrame"""
    return ports.map(process_port)

def shift_and_log(data, shift=DEFAULT_RATE_SHIFT):
    """Apply log transformation with shift to avoid zero values"""
    return np.log10(data + shift)


class CombinedFeatureAdder(BaseEstimator, TransformerMixin):
    """Add combined features for rate ratios"""
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

