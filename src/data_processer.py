import numpy as np
import pandas as pd
import mmh3


DENSE_FEATURES = [f"I{i}" for i in range(1, 14)]
SPARSE_FEATURES = [f"C{i}" for i in range(1, 27)]
LABEL_COL = "label"


def get_criteo_columns(has_label=True):
    """返回 Criteo 数据集列名"""
    features = DENSE_FEATURES + SPARSE_FEATURES
    return [LABEL_COL] + features if has_label else features


def load_criteo_data(file_path, has_label=True):
    """加载 train/test。test.txt 不包含 label 列"""
    return pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=get_criteo_columns(has_label=has_label),
    )


class CriteoPreprocessor:

    def __init__(self, num_bins=10, hash_dim=2**20):
        self.num_bins = num_bins
        self.hash_dim = hash_dim
        self.dense_bin_edges = {}

    def _prepare(self, df, has_label=True):
        df = df.copy()
        
        # dense 
        for feat in DENSE_FEATURES:
            df[feat] = pd.to_numeric(df[feat] , errors="coerce").fillna(0)
        
        # sparse
        for feat in SPARSE_FEATURES:
            df[feat] = df[feat].fillna("-1").astype(str)

        if has_label:
            df[LABEL_COL] = df[LABEL_COL].astype(int)
            
        return df
    
    def fit_dense(self, df):
        quantiles = np.linspace(0, 1, self.num_bins + 1)
        
        for feat in DENSE_FEATURES:
            edges = np.unique(df[feat].quantile(quantiles).values)
            
            if len(edges) <= 1:
                self.dense_bin_edges[feat] = None
                continue
            
            edges[0] = -np.inf
            edges[-1] = np.inf
            self.dense_bin_edges[feat] = edges
            
    def transform_dense(self, df):
        dense_out = []
        
        for feat in DENSE_FEATURES:
            edges = self.dense_bin_edges.get(feat)
            
            if edges is None:
                dense_out.append(np.zeros(len(df)))
                continue
            
            bucket = pd.cut(
                df[feat],
                bins=edges,
                labels=False,
                include_lowest=True,
            ).fillna(0)
            
            dense_out.append(bucket.astype(np.float32))
        
        return np.stack(dense_out, axis=1)
        
    def _hash(self, feat, value):
        return mmh3.hash(f"{feat}={value}") % self.hash_dim
    
    def transform_spare(self, df):
        spare_out = np.zeros((len(df), len(SPARSE_FEATURES)), dtype=np.int64)
        
        for i, feat in enumerate(SPARSE_FEATURES):
            spare_out[:, i] = df[feat].apply(lambda x: self._hash(feat, x)).values
            
        return spare_out
    
    def fit(self, df):
        df = self._prepare(df)
        self.fit_dense(df)
        
    def transform(self, df, has_label=True):
        df = self._prepare(df, has_label)
        
        dense = self.transform_dense(df)
        spare = self.transform_spare(df)
        
        if has_label:
            label = df[LABEL_COL].values.astype(np.float32)
            return dense, spare, label
        
        else:
            return dense, spare