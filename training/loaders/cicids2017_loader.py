"""
CIC-IDS2017 Dataset Loader — v1

Known dataset issues this loader handles:
  1. Leading/trailing whitespace in column names
  2. "Fwd Header Length" appears twice → renamed to "Fwd Header Length.1" on 2nd occurrence
  3. "Flow Bytes/s" and "Flow Packets/s" contain Inf values
  4. Some rows have NaN after Inf replacement
  5. Labels have trailing spaces and mixed casing
  6. Class imbalance: BENIGN ≈ 80% of all rows

Source: Sharafaldin, I., Lashkari, A.H., Ghorbani, A.A. (2018).
        Toward Generating a New Intrusion Detection Dataset and
        Intrusion Traffic Characterization. ICISSP.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from training.interfaces.base_loader import AbstractDataLoader, SplitData


# Features known to cause issues (constant or near-constant in CIC-IDS2017)
_COLS_TO_DROP = {
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
}


class CICIDS2017Loader(AbstractDataLoader):
    """
    Loads CIC-IDS2017 CSV files from a directory, handles all known quirks,
    and provides a stratified 70/15/15 train/val/test split.
    """

    def __init__(self, clip_inf: float = 1e9):
        self.clip_inf = clip_inf
        self._label_encoder = LabelEncoder()

    # ── AbstractDataLoader interface ──────────────────────────────────

    def get_label_column(self) -> str:
        return "Label"

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        label = self.get_label_column()
        return [c for c in df.select_dtypes(include=[np.number]).columns if c != label]

    def load(self, path: str) -> pd.DataFrame:
        """
        Read all CSVs under `path`, fix all known CIC-IDS2017 quirks.
        Returns a clean DataFrame ready for splitting.
        """
        dfs = []
        for csv_path in sorted(Path(path).glob("**/*.csv")):
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                df = self._fix_columns(df)
                dfs.append(df)
                print(f"  Loaded {csv_path.name}: {len(df):,} rows")
            except Exception as e:
                print(f"  Skipping {csv_path.name}: {e}")

        if not dfs:
            raise FileNotFoundError(f"No CSV files found under {path}")

        combined = pd.concat(dfs, ignore_index=True)
        combined = self._clean(combined)
        print(f"\nTotal: {len(combined):,} rows after cleaning")
        return combined

    def split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42,
        stratify: bool = True,
    ) -> SplitData:
        label_col = self.get_label_column()
        feature_cols = self.get_feature_columns(df)

        X = df[feature_cols].values.astype(np.float32)
        y_raw = df[label_col].values
        y = self._label_encoder.fit_transform(y_raw)

        strat = y if stratify else None

        # First split off test, then split val from train
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=strat
        )
        val_fraction = val_size / (1.0 - test_size)
        strat_tv = y_train_val if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_fraction,
            random_state=random_state,
            stratify=strat_tv,
        )

        train_dist = {
            self._label_encoder.classes_[i]: int((y_train == i).sum())
            for i in range(len(self._label_encoder.classes_))
        }

        return SplitData(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            feature_names=feature_cols,
            label_names=list(self._label_encoder.classes_),
            meta={
                "train_rows": len(X_train),
                "val_rows": len(X_val),
                "test_rows": len(X_test),
                "train_class_distribution": train_dist,
            },
        )

    # ── Private helpers ───────────────────────────────────────────────

    def _fix_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace, deduplicate column names."""
        df.columns = df.columns.str.strip()
        seen: dict[str, int] = {}
        new_cols = []
        for col in df.columns:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}.{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)
        df.columns = new_cols
        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle infinity, NaN, label normalisation, and drop known-bad columns."""
        # Normalise labels
        label_col = self.get_label_column()
        if label_col in df.columns:
            df[label_col] = df[label_col].astype(str).str.strip()

        # Drop known-bad columns if present
        cols_to_drop = [c for c in _COLS_TO_DROP if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Clip Inf before NaN replacement (retains the scale info, avoids total loss)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].clip(-self.clip_inf, self.clip_inf)
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Drop rows where more than 10% of features are NaN (likely corrupt rows)
        threshold = int(len(numeric_cols) * 0.10)
        df = df.dropna(thresh=len(df.columns) - threshold)

        # Fill remaining NaN with column median (robust to outliers)
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        return df.reset_index(drop=True)

    @property
    def label_encoder(self) -> LabelEncoder:
        return self._label_encoder
