# features/state_profile.py
from __future__ import annotations
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from data.state_parser import ensure_state, _to_float, _norm_cat

@dataclass
class FieldProfile:
    kind: str
    median: Optional[float] = None
    mad: Optional[float] = None
    q05: Optional[float] = None
    q95: Optional[float] = None
    freq: Dict[str, int] = field(default_factory=dict)
    n: int = 0

class StateProfile:
    def __init__(self, max_auto_categorical_cardinality: int = 20):
        self.max_auto_categorical_cardinality = max_auto_categorical_cardinality
        self.profiles: Dict[str, FieldProfile] = {}

    def fit(self, records: Sequence[Mapping[str, Any]]) -> "StateProfile":
        values: Dict[str, List[Any]] = defaultdict(list)
        for r in records:
            for k, v in ensure_state(r).items():
                values[str(k)].append(v)
        self.profiles = {}
        for k, vals in values.items():
            floats = [_to_float(v) for v in vals]
            valid = np.array([v for v in floats if v is not None], dtype=float)
            cats = [_norm_cat(v) for v in vals if _norm_cat(v) != "<MISSING>"]
            unique_cats = sorted(set(cats))
            if len(unique_cats) <= self.max_auto_categorical_cardinality:
                self.profiles[k] = FieldProfile(kind="categorical", freq=dict(Counter(cats)), n=len(cats))
            elif len(valid) >= max(1, int(0.6 * len(vals))):
                med = float(np.median(valid))
                mad = float(np.median(np.abs(valid - med)))
                q05, q95 = np.quantile(valid, [0.05, 0.95]).tolist()
                self.profiles[k] = FieldProfile(kind="numeric", median=med, mad=mad, q05=float(q05), q95=float(q95), n=len(valid))
            else:
                self.profiles[k] = FieldProfile(kind="categorical", freq=dict(Counter(cats)), n=len(cats))
        return self

    def score_field(self, key: str, value: Any) -> float:
        prof = self.profiles.get(str(key))
        if prof is None:
            return 4.0
        if value is None:
            return 2.0
        if prof.kind == "numeric":
            x = _to_float(value)
            if x is None or prof.median is None:
                return 2.5
            scale = (prof.mad or 0.0) * 1.4826
            if scale <= 1e-9:
                return 5.0 if abs(x - prof.median) > 1e-9 else 0.0
            z = abs(x - prof.median) / scale
            tail_bonus = 1.0 if (prof.q05 is not None and x < prof.q05) or (prof.q95 is not None and x > prof.q95) else 0.0
            return float(min(9.0, z + tail_bonus))
        s = _norm_cat(value)
        n = max(1, prof.n)
        c = prof.freq.get(s, 0)
        if c == 0:
            return 5.0
        return float(max(0.0, -math.log(c / n) - 1.0))

    def top_anomalies(self, record: Mapping[str, Any], exclude_fields: Sequence[str] = (), max_fields: int = 8) -> List[Tuple[str, float]]:
        exclude = set(map(str, exclude_fields))
        scored = []
        for k, v in ensure_state(record).items():
            if str(k) in exclude:
                continue
            score = self.score_field(str(k), v)
            if score > 0.75:
                scored.append((str(k), score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max_fields]