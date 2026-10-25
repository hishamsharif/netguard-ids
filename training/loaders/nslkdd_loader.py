"""
NSL-KDD Dataset Loader — v1

Used for Phase 1 LSTM-AE pretraining on normal network traffic.

Source: Tavallaee, M., et al. (2009). A detailed analysis of the KDD CUP 99
        data set. IEEE Symposium on Computational Intelligence for Security
        and Defense Applications.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from training.interfaces.base_loader import AbstractDataLoader, SplitData


_COLUMN_NAMES = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]
_CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
_DROP_COLS = ["difficulty"]
_NUMERIC_COLS = [
    c for c in _COLUMN_NAMES
    if c not in _CATEGORICAL_COLS + _DROP_COLS + ["label"]
]


class NSLKDDLoader(AbstractDataLoader):
    """
    Loads NSL-KDD train/test files. For LSTM-AE pretraining, only
    normal (benign) flows are used. Categorical columns are ordinally encoded.
    """

    def __init__(self, normal_only: bool = True):
        self.normal_only = normal_only
        self._label_encoder = LabelEncoder()
        self._cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def get_label_column(self) -> str:
        return "label"

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        return [c for c in df.columns if c != "label"]

    def load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, header=None, names=_COLUMN_NAMES)
        df = df.drop(columns=_DROP_COLS)
        df["label"] = df["label"].str.strip().str.lower()

        # Encode categorical features
        df[_CATEGORICAL_COLS] = self._cat_encoder.fit_transform(df[_CATEGORICAL_COLS])

        if self.normal_only:
            df = df[df["label"] == "normal"].copy()
            print(f"  NSL-KDD normal flows: {len(df):,}")

        return df.reset_index(drop=True)

    def split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
        stratify: bool = False,     # single class when normal_only=True
    ) -> SplitData:
        feature_cols = self.get_feature_columns(df)
        X = df[feature_cols].values.astype(np.float32)
        y_raw = df["label"].values
        y = self._label_encoder.fit_transform(y_raw)

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        val_frac = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_frac, random_state=random_state
        )

        return SplitData(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            feature_names=feature_cols,
            label_names=list(self._label_encoder.classes_),
            meta={"train_rows": len(X_train), "val_rows": len(X_val), "test_rows": len(X_test)},
        )
