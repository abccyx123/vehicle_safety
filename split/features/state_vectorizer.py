# features/state_vectorizer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence
import numpy as np
from scipy import sparse
from data.state_parser import ensure_state, _to_float, _norm_cat

@dataclass
class StateFeatureSpec:
    keys: List[str]
    numeric_keys: List[str]
    medians: Dict[str, float]
    categorical_values: Dict[str, List[str]]
    feature_names: List[str]
    max_auto_categorical_cardinality: int = 20

class StateVectorizer:
    def __init__(self, max_auto_categorical_cardinality: int = 20):
        self.max_auto_categorical_cardinality = max_auto_categorical_cardinality
        self.spec: Optional[StateFeatureSpec] = None

    def fit(self, records: Sequence[Mapping[str, Any]]) -> "StateVectorizer":
        key_set = set()
        states = []
        for r in records:
            st = ensure_state(r)
            states.append(st)
            key_set.update(str(k) for k in st.keys())
        keys = sorted(key_set)
        numeric_keys: List[str] = []
        medians: Dict[str, float] = {}
        categorical_values: Dict[str, List[str]] = {}
        feature_names: List[str] = []

        for k in keys:
            vals = [st.get(k) for st in states]
            floats = [_to_float(v) for v in vals]
            valid = [v for v in floats if v is not None]
            if valid and len(valid) >= max(1, int(0.6 * len(vals))):
                numeric_keys.append(k)
                medians[k] = float(np.median(valid))
                feature_names.append(f"num:{k}")
                feature_names.append(f"missing:{k}")
            cats = sorted({_norm_cat(v) for v in vals if _norm_cat(v) != "<MISSING>"})
            if 1 < len(cats) <= self.max_auto_categorical_cardinality:
                categorical_values[k] = cats
                for c in cats:
                    feature_names.append(f"eq:{k}={c}")

        self.spec = StateFeatureSpec(
            keys=keys,
            numeric_keys=numeric_keys,
            medians=medians,
            categorical_values=categorical_values,
            feature_names=feature_names,
            max_auto_categorical_cardinality=self.max_auto_categorical_cardinality,
        )
        return self

    @property
    def feature_names_(self) -> List[str]:
        if self.spec is None:
            raise RuntimeError("StateVectorizer is not fitted")
        return self.spec.feature_names

    def transform(self, records: Sequence[Mapping[str, Any]]) -> sparse.csr_matrix:
        if self.spec is None:
            raise RuntimeError("StateVectorizer is not fitted")
        rows: List[int] = []
        cols: List[int] = []
        data: List[float] = []
        feature_idx = {name: i for i, name in enumerate(self.spec.feature_names)}
        for i, r in enumerate(records):
            st = ensure_state(r)
            for k in self.spec.numeric_keys:
                v = _to_float(st.get(k))
                if v is None:
                    v = self.spec.medians.get(k, 0.0)
                    m = 1.0
                else:
                    m = 0.0
                j = feature_idx[f"num:{k}"]
                rows.append(i); cols.append(j); data.append(float(v))
                if m:
                    j = feature_idx[f"missing:{k}"]
                    rows.append(i); cols.append(j); data.append(m)
            for k, cats in self.spec.categorical_values.items():
                val = _norm_cat(st.get(k))
                if val in cats:
                    j = feature_idx.get(f"eq:{k}={val}")
                    if j is not None:
                        rows.append(i); cols.append(j); data.append(1.0)
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(records), len(self.spec.feature_names)))