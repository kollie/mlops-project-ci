"""
Exploratory Data Analysis (EDA) helper for the MLOps project test suite.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ["EDA"]



def _numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return the names of numeric columns in *df* (helper)."""
    return df.select_dtypes(include=[np.number]).columns.tolist()



class EDA:
   
    def __init__(self, log_file: str = "logs/eda.log", plots_dir: Optional[str] = None):
        self._setup_logging(log_file)

        base_dir = Path(log_file).expanduser().resolve().parent
        self.plots_dir = Path(plots_dir).expanduser().resolve() if plots_dir else base_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.dist_dir = self.plots_dir / "distributions"
        self.dist_dir.mkdir(exist_ok=True)

    # .................................................................. #
    def _setup_logging(self, log_file: str) -> None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure root once – avoids duplicate messages when multiple
        # instances are created inside pytest parametrisations.
        if not logging.getLogger(__name__).handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
                filename=str(log_path),
            )
        self.logger = logging.getLogger(__name__)

    def describe_dataframe(self, df: pd.DataFrame) -> Dict:
        """Return an extended dictionary of descriptive statistics.

        The method returns *empty* structures instead of raising when *df* is
        empty – the tests expect that behaviour.
        """
        if df.empty:
            return {
                "shape": df.shape,
                "dtypes": {},
                "summary": {},
                "memory_usage": {},
                "unique_values": {},
                "skewness": {},
                "kurtosis": {},
            }

        stats: Dict[str, Dict] = {
            "shape": df.shape,
            "dtypes": df.dtypes.astype(str).to_dict(),
            "summary": df.describe(include="all").to_dict(),
            "memory_usage": df.memory_usage(deep=True).astype(int).to_dict(),
            "unique_values": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
        }

        numeric_cols = _numeric_columns(df)
        stats["skewness"] = df[numeric_cols].skew().to_dict()
        stats["kurtosis"] = df[numeric_cols].kurtosis().to_dict()

        self.logger.info("describe_dataframe executed.")
        return stats

    def analyze_target_distribution(self, df: pd.DataFrame, *, target_col: str = "readmitted") -> Dict:
        if df.empty:
            raise ValueError("Cannot analyze target distribution of an empty DataFrame")
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame")

        value_counts = df[target_col].value_counts(dropna=False)
        percentages = value_counts / len(df) * 100

        # Bar chart
        plt.figure(figsize=(10, 6))
        sns.countplot(x=target_col, data=df)
        plt.title(f"Distribution of {target_col}")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "target_distribution.png")
        plt.close()

        # Pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(value_counts, labels=value_counts.index.astype(str), autopct="%1.1f%%")
        plt.title(f"{target_col} Distribution")
        plt.savefig(self.plots_dir / "target_distribution_pie.png")
        plt.close()

        self.logger.info("analyze_target_distribution finished.")
        return {"value_counts": value_counts.to_dict(), "percentages": percentages.to_dict()}

    def analyze_feature_distributions(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        if df.empty:
            return {}

        numeric_cols = _numeric_columns(df)
        if not numeric_cols:
            self.logger.warning("No numerical columns available for distribution analysis.")
            return {}

        stats: Dict[str, Dict[str, float]] = {}
        for col in numeric_cols:
            series = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if series.empty:
                continue

            plt.figure(figsize=(10, 6))
            sns.histplot(series, kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(self.dist_dir / f"{col}_distribution.png")
            plt.close()

            stats[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "skew": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
            }

        self.logger.info("analyze_feature_distributions finished.")
        return stats

    def analyze_correlations(self, df: pd.DataFrame, *, target_col: str = "readmitted") -> pd.DataFrame:
        if df.empty:
            raise ValueError("Cannot analyze correlations of an empty DataFrame")
        numeric_cols = _numeric_columns(df)
        if not numeric_cols:
            raise ValueError("No numerical columns found for correlation analysis")
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame")

        # Create a copy with only numeric columns and target
        helper = df[numeric_cols + [target_col]].copy()
        
        # Convert target to numeric if needed
        if not pd.api.types.is_numeric_dtype(helper[target_col]):
            helper[target_col] = pd.factorize(helper[target_col])[0]

        # Calculate correlations
        corr_matrix = helper.corr()

        # Full heat‑map
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "correlation_heatmap.png")
        plt.close()

        # Top correlations with target (exclude the diagonal)
        target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False).head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=target_corr.values, y=target_corr.index)
        plt.title("Top feature correlations with target")
        plt.xlabel("Correlation coefficient")
        plt.tight_layout()
        plt.savefig(self.plots_dir / "top_correlations.png")
        plt.close()

        self.logger.info("analyze_correlations finished.")
        return corr_matrix

    # ------------------------------------------------------------------ #
    def analyze_class_imbalance(self, df: pd.DataFrame, *, target_col: str = "readmitted") -> Dict:
        if df.empty:
            raise ValueError("Cannot analyze class imbalance of an empty DataFrame")
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame")

        counts = df[target_col].value_counts(dropna=False)
        ratios = counts / len(df)
        imbalance_ratio = float(counts.max() / max(counts.min(), 1))

        results = {
            "class_counts": counts.astype(int).to_dict(),
            "class_ratios": ratios.to_dict(),
            "n_classes": int(counts.size),
            "imbalance_ratio": imbalance_ratio,
        }
        self.logger.info("analyze_class_imbalance finished.")
        return results

    def find_missing_values(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Find missing values in the DataFrame."""
        if df.empty:
            return {"nan_values": {}, "question_mark_values": {}}

        # Find actual NaN values
        nan_counts = df.isna().sum()
        nan_cols = nan_counts[nan_counts > 0].astype(int)
        
        # Find '?' placeholders
        q_cols = {}
        for col in df.columns:
            # Check for exact '?' values
            q_count = (df[col].astype(str) == "?").sum()
            if q_count > 0:
                q_cols[col] = int(q_count)

        missing_values = {
            "nan_values": nan_cols.to_dict(),
            "question_mark_values": q_cols
        }
        
        self.logger.info(f"Missing value analysis:\n{missing_values}")
        return missing_values

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        df = df.copy().replace("?", np.nan)

        # Numeric columns
        for col in _numeric_columns(df):
            median = df[col].median()
            df[col] = df[col].fillna(0 if pd.isna(median) else median)

        # Categorical / object columns
        for col in df.select_dtypes(exclude=[np.number]).columns:
            if df[col].isna().any():
                mode_series = df[col].mode(dropna=True)
                replacement = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                df[col] = df[col].fillna(replacement)

        self.logger.info("handle_missing_values finished.")
        return df

    def remove_low_importance_columns(
        self, df: pd.DataFrame, *, target_col: str = "readmitted", threshold: float = 0.1
    ) -> Tuple[pd.DataFrame, List[str]]:
        if df.empty:
            raise ValueError("Cannot remove columns from an empty DataFrame")
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame")

        numeric_cols = _numeric_columns(df)
        if not numeric_cols:
            self.logger.warning("No numerical columns available for importance analysis.")
            return df.copy(), []

        helper = df[numeric_cols + [target_col]].copy()
        if not pd.api.types.is_numeric_dtype(helper[target_col]):
            helper[target_col] = pd.factorize(helper[target_col])[0]

        correlations = helper[numeric_cols].corrwith(helper[target_col]).abs()
        important_cols = correlations[correlations > threshold].index.tolist()

        cols_to_keep = important_cols + [target_col]
        removed_cols = [c for c in df.columns if c not in cols_to_keep]

        self.logger.info("remove_low_importance_columns finished | %d removed", len(removed_cols))
        return df[cols_to_keep].copy(), removed_cols

    def run_analysis(self, df: pd.DataFrame, *, target_col: str = "readmitted") -> pd.DataFrame:
        if df.empty:
            raise ValueError("Cannot run analysis on an empty DataFrame")

        self.logger.info("---- EDA pipeline start ----")

        # 1. Summary – errors here are unexpected, so let them surface.
        self.describe_dataframe(df)

        # 2. Target distribution – expected to work if target exists.
        self.analyze_target_distribution(df, target_col=target_col)

        # 3. Feature distributions – safe for any df (returns early for no numerics).
        self.analyze_feature_distributions(df)

        # 4. Correlations – may raise if no numeric columns.  Catch + log so the
        #    rest of the pipeline still executes (tests expect that).
        try:
            self.analyze_correlations(df, target_col=target_col)
        except (ValueError, KeyError) as exc:
            self.logger.warning("Correlation analysis skipped: %s", exc)

        # 5. Class imbalance – safe and always computed.
        self.analyze_class_imbalance(df, target_col=target_col)

        # 6. Missing‑value handling – returns cleaned frame.
        df_processed = self.handle_missing_values(df)

        # 7. Remove low‑importance numeric features – never fatal.
        try:
            df_processed, _ = self.remove_low_importance_columns(
                df_processed, target_col=target_col
            )
        except (ValueError, KeyError) as exc:
            self.logger.warning("Low‑importance pruning skipped: %s", exc)

        self.logger.info("---- EDA pipeline end ----")
        return df_processed.copy()
