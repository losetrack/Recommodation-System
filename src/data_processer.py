import numpy as np
import pandas as pd


DENSE_FEATURES = [f"I{i}" for i in range(1, 14)]
SPARSE_FEATURES = [f"C{i}" for i in range(1, 27)]
LABEL_COL = "label"


def get_criteo_columns(has_label=True):
    """返回 Criteo 数据集列名。"""
    features = DENSE_FEATURES + SPARSE_FEATURES
    return [LABEL_COL] + features if has_label else features


def load_criteo_data(file_path, has_label=True):
    """按 readme 规范加载 train/test。test.txt 不包含 label 列。"""
    return pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=get_criteo_columns(has_label=has_label),
    )


class CriteoPreprocessor:
    """训练集 fit，测试集 transform，保证映射一致。"""

    def __init__(self, num_bins=10):
        self.num_bins = num_bins
        self.dense_bin_edges = {}
        self.category_maps = {}
        self.feature_vocab_sizes = {}
        self.is_fitted = False

    def _validate_columns(self, df, has_label=True):
        required_cols = get_criteo_columns(has_label=has_label)
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列: {missing_cols}")

    def _prepare_common(self, df):
        work_df = df.copy()

        for feat in DENSE_FEATURES:
            work_df[feat] = pd.to_numeric(work_df[feat], errors="coerce").fillna(0)

        for feat in SPARSE_FEATURES:
            work_df[feat] = work_df[feat].fillna("-1").astype(str)

        return work_df

    def _fit_dense_bins(self, work_df):
        quantiles = np.linspace(0.0, 1.0, self.num_bins + 1)

        for feat in DENSE_FEATURES:
            series = work_df[feat]
            edges = np.unique(series.quantile(quantiles).to_numpy())

            if len(edges) <= 1:
                self.dense_bin_edges[feat] = None
                continue

            edges = edges.astype(float)
            edges[0] = -np.inf
            edges[-1] = np.inf
            self.dense_bin_edges[feat] = edges.tolist()

    def _bucketize_dense(self, work_df):
        for feat in DENSE_FEATURES:
            edges = self.dense_bin_edges.get(feat)

            if edges is None:
                work_df[feat] = "0"
                continue

            bucket_ids = pd.cut(
                work_df[feat],
                bins=np.array(edges, dtype=float),
                labels=False,
                include_lowest=True,
            )
            work_df[feat] = bucket_ids.fillna(0).astype(int).astype(str)

        return work_df

    def _fit_category_maps(self, work_df):
        for feat in DENSE_FEATURES + SPARSE_FEATURES:
            unique_values = pd.Index(work_df[feat].astype(str).unique())
            # 0 预留给未知类别，已知类别从 1 开始。
            mapping = {v: i + 1 for i, v in enumerate(unique_values)}
            self.category_maps[feat] = mapping
            # 保持与现有模型兼容：vocab_size 仅统计已知类别个数。
            self.feature_vocab_sizes[feat] = len(mapping)

    def _encode_categories(self, work_df):
        for feat in DENSE_FEATURES + SPARSE_FEATURES:
            mapping = self.category_maps[feat]
            work_df[feat] = work_df[feat].astype(str).map(mapping).fillna(0).astype("int64")

        return work_df

    def fit_transform(self, train_df):
        """在训练集上拟合规则并完成转换。"""
        self._validate_columns(train_df, has_label=True)

        work_df = self._prepare_common(train_df)
        self._fit_dense_bins(work_df)
        work_df = self._bucketize_dense(work_df)
        self._fit_category_maps(work_df)
        work_df = self._encode_categories(work_df)

        self.is_fitted = True
        return work_df, self.feature_vocab_sizes.copy()

    def transform(self, df, has_label=False):
        """用训练集规则转换数据，禁止重新拟合。"""
        if not self.is_fitted:
            raise RuntimeError("请先调用 fit_transform 再调用 transform。")

        self._validate_columns(df, has_label=has_label)

        work_df = self._prepare_common(df)
        work_df = self._bucketize_dense(work_df)
        work_df = self._encode_categories(work_df)
        return work_df


def perform_feature_engineering(train_df):
    """兼容旧接口：仅对训练集进行 fit+transform。"""
    preprocessor = CriteoPreprocessor(num_bins=10)
    processed_df, vocab_sizes = preprocessor.fit_transform(train_df)
    return processed_df, vocab_sizes, preprocessor