import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from .utils import *


class BaseModelPipeline:
    """Base class for model pipelines"""
    def __init__(self, name: str, with_pre: bool | None, has_transformer: bool | None = None, clustering: bool = False) -> None:
        self.name = name
        self.with_pre = with_pre
        self.model = None  # Will be loaded separately
        self.pipeline_ = None
        self.transformer_ = has_transformer
        self.is_clustering_ = clustering
        self.cluster_mapping_ = None

    def init_cluster_mapping(self, X, y):
        """Loads/Saves mapping from cluster to category"""
        if not self.is_clustering_:
            raise ValueError('Can only use init_cluster_mapping with clustering algorithms')
        if self.cluster_mapping_ is not None:
            return self
        df = X.copy()
        df['cluster'] = self.predict(X)
        df['category'] = y
        pivot = (
            df
            .groupby(['cluster', 'category'])
            .size()
            .groupby(level=0)
            .apply(lambda x: 100 * x / x.sum())
            .unstack(fill_value=0)
        )
        pivot['top'] = pivot.idxmax(axis=1)
        out = {}
        for i in pivot.index:
            out[i[0]] = pivot.loc[i,'top']
        self.cluster_mapping_ = out
        return self

    def fit_transform(self, X):
        """Transforms input data to proper format"""
        pass

    def pipeline(self):
        """Return complete pipeline with preprocessing"""
        return None

    def category(self, encoded):
        """Returns predictions as category (without encoding)"""
        if self.is_clustering_:
            if self.cluster_mapping_ is not None:
                return np.array(list(map(lambda e: self.cluster_mapping_[e], encoded)))
        return ''

    def predict_label(self, X):
        """Returns predictions as category, not a number"""
        return self.category(self.predict(X))

    def encode_label(self, y):
        """Encodes label to a number as per model definition"""
        pass

    def predict(self, X):
        """Output integer prediction or final pipeline prediction"""
        if self.pipeline_ is None:
            self.pipeline_ = self.pipeline()
            if self.pipeline_ is not None:
                return self.pipeline_.predict(X)
            if self.transformer_ is not None:
                X_prep = self.fit_transform(X)
                return self.model.predict(X_prep)
            else:
                raise ValueError('Either pipeline or transformer function need to be defined and return a non-None value.')
        else:
            return self.pipeline_.predict(X)

class RandomForestNoPreprocPipeline(BaseModelPipeline):
    """Random Forest without preprocessing pipeline"""
    def __init__(self) -> None:
        self.categories_map = {
            'DDoS UDP': 0, 'DDoS TCP': 1, 'DoS UDP': 2, 'DoS TCP': 3, 
            'Reconnaissance OS_Fingerprint': 4, 'Reconnaissance Service_Scan': 5, 
            'Normal': 6, 'Theft': 7
        }
        self.encoder_ = None
        super().__init__('random_forest', False, True)

    def fit_transform(self, X):
        out = X.drop(['saddr', 'daddr', 'seq'], axis=1)
        if self.encoder_ is None:
            print('Hello there')
            # Need to load encoder separately
            raise ValueError('Encoder not loaded')
        feature_cat_cols = ["sport", "dport", "proto"]
        out[feature_cat_cols] = self.encoder_.transform(out[feature_cat_cols].astype(str))
        return out[self.model.feature_names_in_]

    def encode_label(self, y):
        if 'category' in y.columns:
            return y['category'].map(lambda l: self.categories_map[l])

    def category(self, encoded):
        return np.array(list(map(lambda e: [k for k in self.categories_map if self.categories_map[k] == e][0], encoded)))


class XGBoostNoPreprocPipeline(BaseModelPipeline):
  def __init__(self) -> None:
    self.categories = ['DDoS TCP', 'DDoS UDP', 'DoS TCP', 'DoS UDP', 'Normal',
       'Reconnaissance OS_Fingerprint', 'Reconnaissance Service_Scan',
       'Theft']
    self.encoder_ = None
    super().__init__('xgboost_model_final', False, True)

  def fit_transform(self, X: pd.DataFrame):
    cols_processed = [
        'proto', 'sport', 'dport', 'state_number',
        'mean', 'stddev', 'min', 'max', 'srate', 'drate',
        'N_IN_Conn_P_SrcIP', 'N_IN_Conn_P_DstIP'
    ]
    out = X[cols_processed].copy()
    for col in ['sport', 'dport']:
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0)

    if self.encoder_ is None:
      self.encoder_ = ColumnTransformer([
          ('encode', OneHotEncoder(drop='first'), ['proto'])
      ], remainder='passthrough').fit(out)
    # out = pd.get_dummies(out, columns=['proto'], drop_first=True)
    out = self.encoder_.transform(out)
    return out

  def encode_label(self, y):
    if 'category' in y.columns:
      return y['category'].map(lambda l: self.categories.index(l))

  def category(self, encoded):
    return np.array(list(map(lambda e: self.categories[e], encoded)))

class KNNNoPreprocPipeline(BaseModelPipeline):
    """KNN without preprocessing pipeline"""
    def __init__(self) -> None:
        super().__init__('knn', False, True)

    def fit_transform(self, X: pd.DataFrame):
        from sklearn.compose import ColumnTransformer
        features = list(self.model.feature_names_in_)
        out = X.reset_index(drop=False)
        transformer = ColumnTransformer([
            ('port', FunctionTransformer(func=process_ports), ['sport', 'dport']),
        ], remainder='passthrough')
        out = pd.DataFrame(transformer.fit_transform(out[features]), columns=features)
        return out

    def encode_label(self, y):
        return y


class GMMNoPreprocPipeline(BaseModelPipeline):
    """Gaussian Mixture Model without preprocessing pipeline"""
    def __init__(self) -> None:
        super().__init__('gmm_model', False, True, True)
        self.scaler = None  # Need to load separately

    def fit_transform(self, X: pd.DataFrame):
        out = X.drop(['saddr', 'daddr', 'seq'], axis=1).dropna()[[
            "proto", "sport", "dport",
            "min", "max", "mean", "stddev",
            "state_number",
            "N_IN_Conn_P_SrcIP", "N_IN_Conn_P_DstIP",
            "srate", "drate"
        ]]
        out['sport'] = out['sport'].map(process_port)
        out['dport'] = out['dport'].map(process_port)
        proto_map = {"tcp": 0, "udp": 1, "icmp": 2, "arp": 3, "ipv6-icmp": 4}
        out['proto'] = out['proto'].map(lambda pr: proto_map.get(pr, -1))
        if self.scaler is None:
            raise ValueError('Scaler not loaded')
        out = self.scaler.transform(out)
        return out


class KMeansNoPreprocPipeline(BaseModelPipeline):
    """K-Means without preprocessing pipeline"""
    def __init__(self) -> None:
        super().__init__('kmeans', False, True, True)
        self.scaler = None  # Need to load separately

    def fit_transform(self, X: pd.DataFrame):
        out = X[self.model.feature_names_in_]
        out.loc[:, 'sport'] = out['sport'].map(process_port)
        out.loc[:, 'dport'] = out['dport'].map(process_port)
        return out


class RandomForestPreprocPipeline(BaseModelPipeline):
    """Random Forest with full preprocessing pipeline"""
    def __init__(self) -> None:
        self.categories_map = {
            "DDoS TCP": 0, "DDoS UDP": 1, "DoS TCP": 2, "DoS UDP": 3,
            "Normal": 4, "Reconnaissance OS_Fingerprint": 5,
            "Reconnaissance Service_Scan": 6, "Theft": 7,
        }
        super().__init__('random_forest', True, False)

    def pipeline(self):
        return self.model

    def encode_label(self, y):
        if 'category' in y.columns:
            return y['category'].map(lambda l: self.categories_map[l])
        return None

    def category(self, encoded):
        return np.array(list(map(lambda e: [k for k in self.categories_map if self.categories_map[k] == e][0], encoded)))


class XGBoostPreprocPipeline(BaseModelPipeline):
    """XGBoost with full preprocessing pipeline"""
    def __init__(self) -> None:
        self.categories_map = {
            "DDoS TCP": 0, "DDoS UDP": 1, "DoS TCP": 2, "DoS UDP": 3,
            "Normal": 4, "Reconnaissance OS_Fingerprint": 5,
            "Reconnaissance Service_Scan": 6, "Theft": 7,
        }
        super().__init__('xgboost_classifier', True, False)

    def pipeline(self):
        return self.model

    def encode_label(self, y):
        if 'category' in y.columns:
            return y['category'].map(lambda l: self.categories_map[l])
        return None

    def category(self, encoded):
        return np.array(list(map(lambda e: [k for k in self.categories_map if self.categories_map[k] == e][0], encoded)))


class KNNPreprocPipeline(BaseModelPipeline):
    """KNN with full preprocessing pipeline"""
    def __init__(self) -> None:
        self.categories_map = {
            "DDoS TCP": 0, "DDoS UDP": 1, "DoS TCP": 2, "DoS UDP": 3,
            "Normal": 4, "Reconnaissance OS_Fingerprint": 5,
            "Reconnaissance Service_Scan": 6, "Theft": 7,
        }
        super().__init__('K-neirest-neighbors', True, False)

    def pipeline(self):
        return self.model

    def encode_label(self, y):
        if 'category' in y.columns:
            return y['category'].map(lambda l: self.categories_map[l])
        return None

    def category(self, encoded):
        return np.array(list(map(lambda e: [k for k in self.categories_map if self.categories_map[k] == e][0], encoded)))
