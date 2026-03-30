import numpy as np
import pandas as pd
import mmh3


DENSE_FEATURES = [f"I{i}" for i in range(1, 14)]
SPARSE_FEATURES = [f"C{i}" for i in range(1, 27)]
LABEL_COL = "label"


def get_criteo_columns(has_label=True):
    """返回 Criteo 数据集列名列表。"""
    features = DENSE_FEATURES + SPARSE_FEATURES
    return [LABEL_COL] + features if has_label else features


def load_criteo_data(file_path, has_label=True):
    """按 Criteo 列定义加载 TSV 数据文件。"""
    return pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=get_criteo_columns(has_label=has_label),
    )


class CriteoPreprocessor:

    def __init__(self, num_bins=10, hash_dim=2**20):
        """初始化分箱数量、哈希空间及预处理状态。"""
        self.num_bins = num_bins
        self.hash_dim = hash_dim
        self.dense_bin_edges = {}

    def _prepare(self, df, has_label=True):
        """清洗原始数据并统一 dense/sparse/label 字段类型。"""
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
        """基于分位数拟合连续特征的分箱边界。"""
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
        """将连续特征映射为分箱后的离散索引。"""
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
        """对单个稀疏特征值执行哈希编码。"""
        return mmh3.hash(f"{feat}={value}") % self.hash_dim
    
    def transform_spare(self, df):
        """将稀疏特征批量哈希为定长整数索引矩阵。"""
        spare_out = np.zeros((len(df), len(SPARSE_FEATURES)), dtype=np.int64)
        
        for i, feat in enumerate(SPARSE_FEATURES):
            spare_out[:, i] = df[feat].apply(lambda x: self._hash(feat, x)).values
            
        return spare_out
    
    def fit(self, df):
        """拟合预处理器，仅学习连续特征分箱边界。"""
        df = self._prepare(df)
        self.fit_dense(df)
        
    def transform(self, df, has_label=True):
        """将原始样本转换为模型输入特征与可选标签。"""
        df = self._prepare(df, has_label)
        
        dense = self.transform_dense(df)
        spare = self.transform_spare(df)
        
        if has_label:
            label = df[LABEL_COL].values.astype(np.float32)
            return dense, spare, label
        
        else:
            return dense, spare