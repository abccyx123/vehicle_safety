# -*- coding: utf-8 -*-
"""
Automatic rule mining and problem/answer dataset construction for vehicle-instruction risk RL/SFT.

Design constraints implemented here:
- Features are learned from the input records only. No hand-written risk features,
  no command_family / scene_name / scenario fields, and no hand-coded risk keyword list.
- Rule labels are learned from expected.vehicle_risk and expected.instruction_risk.
- Risk phrases are used only to annotate mined leaves after fitting; they are never
  used as model features.
- Training problem construction can be out-of-fold to reduce label leakage.
- Output records are {"problem": str, "answer": str}; answer is compact JSON text only.

Input JSONL fields used:
  query, state_raw/state, expected.vehicle_risk, expected.instruction_risk,
  expected.all_vehicle_risk_keywords/all_instruction_risk_keywords/risk_tip (optional).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold


RISK_ORDER = {"safe": 0, "medium": 1, "high": 2}
ORDER_RISK = {v: k for k, v in RISK_ORDER.items()}
SEVERITY_WEIGHT = {"safe": 0.82, "medium": 1.08, "high": 1.18}


# ----------------------------- basic IO helpers -----------------------------

def normalize_label(x: Any) -> str:
    s = str(x or "safe").strip().lower()
    if s in {"none", "normal", "low", "0"}:
        return "safe"
    if s in {"mid", "moderate", "1"}:
        return "medium"
    if s in {"severe", "danger", "2"}:
        return "high"
    return s if s in RISK_ORDER else "safe"


def max_risk(a: str, b: str) -> str:
    return ORDER_RISK[max(RISK_ORDER[normalize_label(a)], RISK_ORDER[normalize_label(b)])]


def load_jsonl(path: str | Path, split: Optional[str] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if split is not None and str(r.get("split", "")) != split:
                continue
            r.setdefault("_line_no", line_no)
            out.append(r)
    return out


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def default_parquet_path(jsonl_path: str | Path) -> Path:
    path = Path(jsonl_path)
    if path.suffix.lower() == ".jsonl":
        return path.with_suffix(".parquet")
    if path.suffix.lower() == ".parquet":
        return path
    return Path(str(path) + ".parquet")


def write_parquet(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write the same problem/answer records to Parquet.

    Requires pandas plus a parquet engine such as pyarrow. The repository
    requirements include pyarrow so the dataset writer can emit both required
    formats instead of silently dropping parquet.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError("Writing Parquet requires pandas plus pyarrow/fastparquet") from e
    df = pd.DataFrame(list(records), columns=["problem", "answer"])
    try:
        df.to_parquet(path, index=False, engine="pyarrow")
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError("Writing Parquet failed. Install pyarrow, e.g. `pip install pyarrow`.") from e


def write_dataset_outputs(jsonl_path: str | Path, records: Sequence[Mapping[str, Any]],
                          parquet_path: Optional[str | Path] = None, skip_parquet: bool = False) -> Tuple[Path, Optional[Path]]:
    """Write identical JSONL and Parquet datasets.

    The materialized rows are intentionally limited to {"problem": str, "answer": str}.
    """
    rows = [{"problem": str(r["problem"]), "answer": str(r["answer"])} for r in records]
    jsonl_path = Path(jsonl_path)
    if jsonl_path.suffix.lower() == ".parquet":
        jsonl_path = jsonl_path.with_suffix(".jsonl")
    write_jsonl(jsonl_path, rows)
    if skip_parquet:
        return jsonl_path, None
    pq_path = Path(parquet_path) if parquet_path is not None else default_parquet_path(jsonl_path)
    write_parquet(pq_path, rows)
    return jsonl_path, pq_path


def record_id(record: Mapping[str, Any]) -> Any:
    return record.get("index", record.get("id", record.get("_line_no")))


def ensure_state(record: Mapping[str, Any]) -> Mapping[str, Any]:
    # Prefer raw state because it is the field available to an online model.
    state = record.get("state_raw")
    if isinstance(state, Mapping):
        return state
    state = record.get("state")
    if isinstance(state, Mapping):
        return state
    return {}


def infer_raw_to_readable(record: Mapping[str, Any]) -> Dict[str, str]:
    """Infer raw->readable field names from a single record without any schema.

    The provided dataset writes state_raw and state in the same insertion order.
    If that invariant is absent, this safely falls back to raw names.
    """
    raw = record.get("state_raw") or {}
    readable = record.get("state") or {}
    if not isinstance(raw, Mapping) or not isinstance(readable, Mapping):
        return {}
    raw_keys = list(raw.keys())
    rd_keys = list(readable.keys())
    if len(raw_keys) != len(rd_keys):
        return {}
    return {str(rk): str(rd_keys[i]) for i, rk in enumerate(raw_keys)}


def expected_label(record: Mapping[str, Any], task: str) -> str:
    exp = record.get("expected") or {}
    if task == "vehicle":
        return normalize_label(exp.get("vehicle_risk", "safe"))
    if task == "instruction":
        return normalize_label(exp.get("instruction_risk", "safe"))
    raise ValueError(f"unknown task: {task}")


def task_phrases(record: Mapping[str, Any], task: str) -> List[str]:
    """Collect per-record risk prompt phrases, with no hard-coded vocabulary.

    Existing keyword arrays are used when present because they are labels attached
    to the individual case, not a manually enumerated global dictionary. If they
    are not present, fall back to the case risk_tip as a weak phrase.
    """
    exp = record.get("expected") or {}
    if task == "vehicle":
        label = normalize_label(exp.get("vehicle_risk", "safe"))
        if label == "safe":
            return []
        kws = exp.get("all_vehicle_risk_keywords") or []
    elif task == "instruction":
        label = normalize_label(exp.get("instruction_risk", "safe"))
        if label == "safe":
            return []
        kws = exp.get("all_instruction_risk_keywords") or []
    else:
        raise ValueError(f"unknown task: {task}")
    clean: List[str] = []
    for x in kws:
        s = str(x).strip()
        if s and s not in clean:
            clean.append(s)
    if clean:
        return clean
    tip = str(exp.get("risk_tip", "")).strip()
    return [tip] if tip else []


# ----------------------------- state features ------------------------------

def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return float(int(x))
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            if np.isnan(float(x)):
                return None
        except Exception:
            pass
        return float(x)
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _norm_cat(x: Any) -> str:
    if x is None:
        return "<MISSING>"
    if isinstance(x, float) and abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return str(x).strip()


@dataclass
class StateFeatureSpec:
    keys: List[str]
    numeric_keys: List[str]
    medians: Dict[str, float]
    categorical_values: Dict[str, List[str]]
    feature_names: List[str]
    max_auto_categorical_cardinality: int = 20


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
    """Unsupervised state distribution profile for OOD / novel-risk fallback.

    This profile is learned only from raw state values. It does not encode any
    hand-written risk feature or risk keyword. At inference time it can surface
    unusual fields when the active tree rules are uncertain or too narrow.
    """

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
            # Low-cardinality integer/enum-like fields are treated as categorical
            # for anomaly scoring; continuous fields are scored robustly.
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
                # Constant during training: unseen deviation is informative.
                return 5.0 if abs(x - prof.median) > 1e-9 else 0.0
            z = abs(x - prof.median) / scale
            tail_bonus = 1.0 if (prof.q05 is not None and x < prof.q05) or (prof.q95 is not None and x > prof.q95) else 0.0
            return float(min(9.0, z + tail_bonus))
        s = _norm_cat(value)
        n = max(1, prof.n)
        c = prof.freq.get(s, 0)
        if c == 0:
            return 5.0
        # Rare categorical values get a small positive score; common values are near 0.
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


class StateVectorizer:
    """Schema-free vectorizer for vehicle states.

    It only uses raw columns appearing in the data. Low-cardinality columns are
    automatically one-hot encoded; all numeric columns also keep their numeric
    value plus a missing flag. This is generic preprocessing, not a risk rule.
    """

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
            # Auto-discover categorical/equality tests. Numeric enum-like columns
            # will be represented this way if they have low cardinality.
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
            # Numeric + missing.
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
            # Auto one-hot equality features.
            for k, cats in self.spec.categorical_values.items():
                val = _norm_cat(st.get(k))
                if val in cats:
                    j = feature_idx.get(f"eq:{k}={val}")
                    if j is not None:
                        rows.append(i); cols.append(j); data.append(1.0)
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(records), len(self.spec.feature_names)))


# ---------------------------- combined features ----------------------------

class CombinedFeaturizer:
    def __init__(self, task: str, max_query_features: int = 1200, ngram_min: int = 2, ngram_max: int = 4,
                 min_query_df: int = 2, max_auto_categorical_cardinality: int = 20):
        self.task = task
        self.state_vectorizer = StateVectorizer(max_auto_categorical_cardinality=max_auto_categorical_cardinality)
        self.query_vectorizer: Optional[CountVectorizer] = None
        self.max_query_features = max_query_features
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max
        self.min_query_df = min_query_df
        self.feature_names: List[str] = []

    def fit(self, records: Sequence[Mapping[str, Any]]) -> "CombinedFeaturizer":
        self.state_vectorizer.fit(records)
        state_names = self.state_vectorizer.feature_names_
        if self.task == "vehicle":
            self.feature_names = state_names
            return self
        if self.task != "instruction":
            raise ValueError(f"unknown task: {self.task}")
        self.query_vectorizer = CountVectorizer(
            analyzer="char",
            ngram_range=(self.ngram_min, self.ngram_max),
            binary=True,
            min_df=self.min_query_df,
            max_features=self.max_query_features,
        )
        queries = [str(r.get("query", "")) for r in records]
        self.query_vectorizer.fit(queries)
        q_names = [f"q:{x}" for x in self.query_vectorizer.get_feature_names_out().tolist()]
        self.feature_names = q_names + state_names
        return self

    def transform(self, records: Sequence[Mapping[str, Any]]) -> sparse.csr_matrix:
        Xs = self.state_vectorizer.transform(records)
        if self.task == "vehicle":
            return Xs
        if self.query_vectorizer is None:
            raise RuntimeError("query vectorizer is not fitted")
        Xq = self.query_vectorizer.transform([str(r.get("query", "")) for r in records])
        return sparse.hstack([Xq, Xs], format="csr")

    def transform_one(self, record: Mapping[str, Any]) -> sparse.csr_matrix:
        return self.transform([record])


# ------------------------------- rule model --------------------------------

@dataclass
class RuleCondition:
    feature_index: int
    feature_name: str
    op: str
    threshold: float
    text: str
    raw_field: Optional[str] = None
    query_fragment: Optional[str] = None
    positive: bool = True


@dataclass
class MinedRule:
    rule_id: str
    task: str
    tree_index: int
    leaf_id: int
    label: str
    support: int
    precision: float
    class_counts: Dict[str, int]
    conditions: List[RuleCondition]
    phrases: List[str] = field(default_factory=list)
    exemplar_ids: List[Any] = field(default_factory=list)
    score_base: float = 0.0

    @property
    def state_fields(self) -> List[str]:
        out: List[str] = []
        for c in self.conditions:
            if c.raw_field and c.raw_field not in out:
                out.append(c.raw_field)
        return out

    @property
    def query_fragments(self) -> List[str]:
        out: List[str] = []
        for c in self.conditions:
            if c.query_fragment and c.positive and c.query_fragment not in out:
                out.append(c.query_fragment)
        return out

    def compact_conditions(self, max_conditions: int = 8) -> List[str]:
        # Prefer positive query terms and state conditions; negative query terms are
        # usually less useful for an LLM prompt.
        positives: List[str] = []
        state_or_numeric: List[str] = []
        negatives: List[str] = []
        for c in self.conditions:
            if c.query_fragment and c.positive:
                positives.append(c.text)
            elif c.raw_field:
                state_or_numeric.append(c.text)
            else:
                negatives.append(c.text)
        merged: List[str] = []
        # Negative query conditions are kept only as a tiny fallback; they are
        # often artifacts of tree partitioning and less useful for guiding an LLM.
        for group in (positives, state_or_numeric):
            for x in group:
                if x not in merged:
                    merged.append(x)
                if len(merged) >= max_conditions:
                    return merged
        for x in negatives[:2]:
            if x not in merged:
                merged.append(x)
            if len(merged) >= max_conditions:
                return merged
        return merged


@dataclass
class RetrievalResult:
    task: str
    predicted_label: str
    label_votes: Dict[str, float]
    rules: List[MinedRule]


def _feature_condition_text(feature_name: str, op: str, threshold: float) -> Tuple[str, Optional[str], Optional[str], bool]:
    """Return text, raw field, query fragment, positivity."""
    positive = op == ">"
    if feature_name.startswith("q:"):
        frag = feature_name[2:]
        if threshold <= 0.5:
            if op == ">":
                return f"query 包含『{frag}』", None, frag, True
            return f"query 不包含『{frag}』", None, frag, False
        # CountVectorizer is binary, so thresholds should almost always be 0.5.
        return f"query {'>' if op == '>' else '<='} {threshold:.3g} on 『{frag}』", None, frag, positive
    if feature_name.startswith("eq:"):
        body = feature_name[3:]
        k, _, val = body.partition("=")
        if threshold <= 0.5:
            if op == ">":
                return f"state.{k} == {val}", k, None, True
            return f"state.{k} != {val}", k, None, False
        return f"state.{k} {'>' if op == '>' else '<='} {threshold:.3g} for value {val}", k, None, positive
    if feature_name.startswith("num:"):
        k = feature_name[4:]
        return f"state.{k} {op} {threshold:.4g}", k, None, positive
    if feature_name.startswith("missing:"):
        k = feature_name[8:]
        if threshold <= 0.5:
            if op == ">":
                return f"state.{k} 缺失", k, None, True
            return f"state.{k} 未缺失", k, None, False
        return f"missing({k}) {op} {threshold:.3g}", k, None, positive
    return f"{feature_name} {op} {threshold:.4g}", None, None, positive


def _iter_leaf_paths(tree, feature_names: Sequence[str]) -> Dict[int, List[RuleCondition]]:
    t = tree.tree_
    out: Dict[int, List[RuleCondition]] = {}
    stack: List[Tuple[int, List[RuleCondition]]] = [(0, [])]
    while stack:
        node_id, path = stack.pop()
        left = t.children_left[node_id]
        right = t.children_right[node_id]
        if left == -1 and right == -1:
            out[int(node_id)] = path
            continue
        fi = int(t.feature[node_id])
        thr = float(t.threshold[node_id])
        fname = feature_names[fi] if 0 <= fi < len(feature_names) else f"f{fi}"
        txt, raw_field, qfrag, pos = _feature_condition_text(fname, "<=", thr)
        left_cond = RuleCondition(fi, fname, "<=", thr, txt, raw_field, qfrag, pos)
        txt, raw_field, qfrag, pos = _feature_condition_text(fname, ">", thr)
        right_cond = RuleCondition(fi, fname, ">", thr, txt, raw_field, qfrag, pos)
        stack.append((left, path + [left_cond]))
        stack.append((right, path + [right_cond]))
    return out


class AutoRuleMiner:
    """Random-forest leaf rule miner for one task."""

    def __init__(self, task: str, n_estimators: int = 80, max_depth: int = 8, min_samples_leaf: int = 5,
                 random_state: int = 42, max_query_features: int = 1200, min_query_df: int = 2,
                 max_auto_categorical_cardinality: int = 20):
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.max_query_features = max_query_features
        self.min_query_df = min_query_df
        self.max_auto_categorical_cardinality = max_auto_categorical_cardinality
        self.featurizer = CombinedFeaturizer(
            task=task,
            max_query_features=max_query_features,
            min_query_df=min_query_df,
            max_auto_categorical_cardinality=max_auto_categorical_cardinality,
        )
        self.model: Optional[RandomForestClassifier] = None
        self.rules: List[MinedRule] = []
        self.rule_by_tree_leaf: Dict[Tuple[int, int], MinedRule] = {}
        self.train_summary: Dict[str, Any] = {}

    def fit(self, records: Sequence[Mapping[str, Any]]) -> "AutoRuleMiner":
        if not records:
            raise ValueError("AutoRuleMiner.fit got empty records")
        self.featurizer.fit(records)
        X = self.featurizer.transform(records)
        y = np.array([expected_label(r, self.task) for r in records], dtype=object)
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=max(2, self.min_samples_leaf * 2),
            class_weight="balanced_subsample",
            max_features="sqrt",
            random_state=self.random_state,
            n_jobs=1,
        )
        self.model.fit(X, y)
        self._extract_rules(records, X, y)
        self.train_summary = {
            "task": self.task,
            "n_train": len(records),
            "label_counts": dict(Counter(y.tolist())),
            "n_rules": len(self.rules),
            "feature_count": len(self.featurizer.feature_names),
            "model": "RandomForestClassifier",
        }
        return self

    def _extract_rules(self, records: Sequence[Mapping[str, Any]], X: sparse.csr_matrix, y: np.ndarray) -> None:
        if self.model is None:
            raise RuntimeError("model not fitted")
        leaf_mat = self.model.apply(X)
        feature_names = self.featurizer.feature_names
        self.rules = []
        self.rule_by_tree_leaf = {}
        for tree_idx, tree in enumerate(self.model.estimators_):
            paths = _iter_leaf_paths(tree, feature_names)
            leaf_to_indices: Dict[int, List[int]] = defaultdict(list)
            for row_idx, leaf_id in enumerate(leaf_mat[:, tree_idx].tolist()):
                leaf_to_indices[int(leaf_id)].append(row_idx)
            for leaf_id, idxs in leaf_to_indices.items():
                if not idxs:
                    continue
                labels = [str(y[i]) for i in idxs]
                counts = Counter(labels)
                label, cnt = counts.most_common(1)[0]
                support = len(idxs)
                precision = float(cnt / max(1, support))
                phrase_counter: Counter[str] = Counter()
                for i in idxs:
                    # Only phrases compatible with this rule's task and predicted label.
                    if expected_label(records[i], self.task) == label:
                        for p in task_phrases(records[i], self.task):
                            phrase_counter[p] += 1
                phrases = [p for p, _ in phrase_counter.most_common(5)]
                exemplar_ids = [record_id(records[i]) for i in idxs[:3]]
                conds = paths.get(int(leaf_id), [])
                # A compact, low-support rule is better than a very long precise one.
                depth_penalty = 1.0 / (1.0 + 0.07 * max(0, len(conds) - 5))
                score_base = precision * math.log1p(support) * SEVERITY_WEIGHT.get(label, 1.0) * depth_penalty
                rid = f"{self.task}_t{tree_idx}_l{leaf_id}"
                rule = MinedRule(
                    rule_id=rid,
                    task=self.task,
                    tree_index=tree_idx,
                    leaf_id=int(leaf_id),
                    label=normalize_label(label),
                    support=int(support),
                    precision=precision,
                    class_counts={k: int(v) for k, v in counts.items()},
                    conditions=conds,
                    phrases=phrases,
                    exemplar_ids=exemplar_ids,
                    score_base=float(score_base),
                )
                self.rules.append(rule)
                self.rule_by_tree_leaf[(tree_idx, int(leaf_id))] = rule

    def predict_proba_label_votes(self, X: sparse.csr_matrix) -> Dict[str, float]:
        if self.model is None:
            raise RuntimeError("model not fitted")
        out = {"safe": 0.0, "medium": 0.0, "high": 0.0}
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            for cls, p in zip(self.model.classes_, proba):
                out[normalize_label(cls)] = float(p)
        else:
            pred = normalize_label(self.model.predict(X)[0])
            out[pred] = 1.0
        return out

    def retrieve(self, record: Mapping[str, Any], top_k: int = 6, min_precision: float = 0.0,
                 prefer_non_safe: bool = True) -> RetrievalResult:
        if self.model is None:
            raise RuntimeError("model not fitted")
        X = self.featurizer.transform_one(record)
        leaf_ids = self.model.apply(X)[0].tolist()
        votes = self.predict_proba_label_votes(X)
        predicted_label = max(votes.items(), key=lambda kv: kv[1])[0]
        candidates: List[MinedRule] = []
        for tree_idx, leaf_id in enumerate(leaf_ids):
            rule = self.rule_by_tree_leaf.get((tree_idx, int(leaf_id)))
            if rule and rule.precision >= min_precision:
                candidates.append(rule)
        # De-duplicate very similar rules by signature while retaining highest score.
        best_by_sig: Dict[Tuple[str, Tuple[str, ...]], MinedRule] = {}
        for r in candidates:
            cond_sig = tuple(r.compact_conditions(max_conditions=6))
            sig = (r.label, cond_sig)
            existing = best_by_sig.get(sig)
            if existing is None or r.score_base > existing.score_base:
                best_by_sig[sig] = r
        cand = list(best_by_sig.values())
        def rank(rule: MinedRule) -> float:
            pred_bonus = 1.15 if rule.label == predicted_label else 1.0
            # For prompt construction, non-safe rules are much more useful than
            # broad safe leaves full of negative query conditions. Safe evidence is
            # kept only as a small fallback when no risk rule is active.
            non_safe_bonus = 1.35 if prefer_non_safe and rule.label != "safe" else 0.62
            lift = rule.precision / max(1e-6, votes.get(rule.label, 1e-6))
            return rule.score_base * pred_bonus * non_safe_bonus * min(2.0, max(0.6, lift))
        non_safe = [r for r in cand if r.label != "safe"]
        safe = [r for r in cand if r.label == "safe"]
        non_safe.sort(key=rank, reverse=True)
        safe.sort(key=rank, reverse=True)
        if non_safe:
            selected = non_safe[:top_k]
            # Include at most one safe counter-prior if the vote is safe or near-safe.
            if predicted_label == "safe" and len(selected) < top_k and safe:
                selected.append(safe[0])
        else:
            selected = safe[:min(1, top_k)]
        return RetrievalResult(task=self.task, predicted_label=predicted_label, label_votes=votes, rules=selected[:top_k])

    def rules_as_jsonable(self, max_conditions: int = 20) -> List[Dict[str, Any]]:
        out = []
        for r in self.rules:
            out.append({
                "rule_id": r.rule_id,
                "task": r.task,
                "tree_index": r.tree_index,
                "leaf_id": r.leaf_id,
                "label": r.label,
                "support": r.support,
                "precision": round(r.precision, 6),
                "class_counts": r.class_counts,
                "conditions": [asdict(c) for c in r.conditions[:max_conditions]],
                "phrases": r.phrases,
                "exemplar_ids": r.exemplar_ids,
                "score_base": round(r.score_base, 6),
            })
        return out


class DualRiskRuleBank:
    def __init__(self, vehicle_miner: AutoRuleMiner, instruction_miner: AutoRuleMiner,
                 metadata: Optional[Dict[str, Any]] = None, state_profile: Optional[StateProfile] = None):
        self.vehicle_miner = vehicle_miner
        self.instruction_miner = instruction_miner
        self.metadata = metadata or {}
        self.state_profile = state_profile

    @classmethod
    def fit(cls, records: Sequence[Mapping[str, Any]], n_estimators: int = 80, max_depth: int = 8,
            min_samples_leaf: int = 5, random_state: int = 42, max_query_features: int = 1200,
            min_query_df: int = 2, max_auto_categorical_cardinality: int = 20) -> "DualRiskRuleBank":
        vehicle = AutoRuleMiner(
            "vehicle", n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            random_state=random_state, max_query_features=max_query_features, min_query_df=min_query_df,
            max_auto_categorical_cardinality=max_auto_categorical_cardinality,
        ).fit(records)
        instruction = AutoRuleMiner(
            "instruction", n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
            random_state=random_state + 1009, max_query_features=max_query_features, min_query_df=min_query_df,
            max_auto_categorical_cardinality=max_auto_categorical_cardinality,
        ).fit(records)
        state_profile = StateProfile(max_auto_categorical_cardinality=max_auto_categorical_cardinality).fit(records)
        md = {
            "n_train": len(records),
            "random_state": random_state,
            "uses_fields": ["query", "state_raw/state", "expected labels and per-case phrases for leaf annotation only"],
            "excludes_fields": ["command_family", "scene_name", "vehicle_scenario_type", "instruction_scenario_type"],
            "vehicle_summary": vehicle.train_summary,
            "instruction_summary": instruction.train_summary,
            "state_profile": "unsupervised robust numeric/categorical rarity profile",
        }
        return cls(vehicle, instruction, md, state_profile=state_profile)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "DualRiskRuleBank":
        return joblib.load(path)

    def retrieve(self, record: Mapping[str, Any], top_k_vehicle: int = 5, top_k_instruction: int = 5,
                 min_precision: float = 0.0) -> Dict[str, RetrievalResult]:
        return {
            "vehicle": self.vehicle_miner.retrieve(record, top_k=top_k_vehicle, min_precision=min_precision),
            "instruction": self.instruction_miner.retrieve(record, top_k=top_k_instruction, min_precision=min_precision),
        }

    def export_rules(self, path: str | Path, max_rules_per_task: int = 5000) -> None:
        payload = {
            "metadata": self.metadata,
            "vehicle_rules": self.vehicle_miner.rules_as_jsonable()[:max_rules_per_task],
            "instruction_rules": self.instruction_miner.rules_as_jsonable()[:max_rules_per_task],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------- dataset construction ----------------------------

def _short_value(value: Any, max_len: int = 40) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating,)):
        x = float(value)
        if math.isnan(x):
            return ""
        return f"{x:.6g}"
    v = str(value).strip()
    return v if len(v) <= max_len else v[:max_len] + "…"


def _display_state_name_value(record: Mapping[str, Any], raw_field: str) -> Tuple[str, str]:
    """Map one raw state id to a prompt-side name:text pair.

    The mapping is inferred from the example's own `state_raw`/`state` order, so
    this remains schema-free and does not introduce hand-written risk features.
    """
    raw = ensure_state(record)
    readable_map = infer_raw_to_readable(record)
    readable_state = record.get("state") if isinstance(record.get("state"), Mapping) else {}
    rd = readable_map.get(raw_field)
    name = str(rd or raw_field)
    value = readable_state.get(rd) if rd and isinstance(readable_state, Mapping) else raw.get(raw_field)
    return name, _short_value(value)


def _all_state_name_text(record: Mapping[str, Any], max_state_fields: int = 512) -> Dict[str, str]:
    """Return the complete current vehicle state as {readable_name: text_value}.

    This is the OOD fallback: even if no mined rule covers a novel risk, the model
    still sees every state signal in the same compact name:text representation.
    """
    out: Dict[str, str] = {}
    readable_state = record.get("state") if isinstance(record.get("state"), Mapping) else None
    if readable_state:
        for k, v in list(readable_state.items())[:max_state_fields]:
            name = str(k)
            if name in out:
                name = f"{name}#{len(out)}"
            out[name] = _short_value(v)
        return out

    raw = ensure_state(record)
    for k in list(raw.keys())[:max_state_fields]:
        name, value = _display_state_name_value(record, str(k))
        if name in out:
            name = f"{name}({k})"
        out[name] = value
    return out


def _state_field_names(record: Mapping[str, Any], fields: Sequence[str], max_fields: Optional[int] = None) -> List[str]:
    out: List[str] = []
    seq = fields[:max_fields] if max_fields is not None else fields
    for f in seq:
        name, _ = _display_state_name_value(record, str(f))
        if name not in out:
            out.append(name)
    return out


def _rule_evidence(rule: MinedRule, record: Optional[Mapping[str, Any]] = None, max_conditions: int = 5) -> List[str]:
    """Readable evidence currently activated by a mined tree leaf."""
    conds: List[str] = []
    for x in rule.compact_conditions(max_conditions=max_conditions + 4):
        if x.startswith("query 包含") and x not in conds:
            conds.append(x)
    if record is not None:
        for f in rule.state_fields:
            name, value = _display_state_name_value(record, f)
            item = f"{name}={value}"
            if item not in conds:
                conds.append(item)
            if len(conds) >= max_conditions:
                break
    if not conds:
        # Keep positive/non-query conditions first; avoid broad "query 不包含" unless
        # there is literally no other leaf evidence.
        conds = [x for x in rule.compact_conditions(max_conditions=max_conditions + 2) if not x.startswith("query 不包含")]
    return (conds or rule.compact_conditions(max_conditions=1))[:max_conditions]


def _rule_to_understanding(rule: MinedRule, record: Optional[Mapping[str, Any]] = None,
                           max_conditions: int = 5, include_phrases: bool = False) -> Dict[str, Any]:
    """Return the tree leaf's learned understanding without probabilities.

    v6 keeps the useful part: the tree's learned risk direction and the activated
    evidence. It intentionally omits probability, precision and support so a small
    LLM is less likely to copy a wrong forest confidence. The label inside tree_understanding is the
    tree leaf's hypothesis, not a final label.
    """
    label_name = "vehicle_risk" if rule.task == "vehicle" else "instruction_risk"
    d: Dict[str, Any] = {
        "tree_understanding": f"这些命中证据在训练树叶中主要对应 {label_name}={rule.label}；这只是候选假设，可被当前完整状态和指令推翻。",
        "evidence": _rule_evidence(rule, record=record, max_conditions=max_conditions),
    }
    if include_phrases and rule.phrases:
        # Off by default. This is intended only for debugging ablations because it
        # can leak an implicit enumerated risk dictionary into the prompt.
        d["phrases"] = rule.phrases[:2]
    return d


def _unique_state_fields_from_rules(rules: Sequence[MinedRule], max_fields: Optional[int] = None) -> List[str]:
    fields: List[str] = []
    for r in rules:
        for f in r.state_fields:
            if f not in fields:
                fields.append(f)
    if max_fields is not None:
        fields = fields[:max_fields]
    return fields


def build_full_state_context(record: Mapping[str, Any], retrieval: Mapping[str, RetrievalResult],
                             state_profile: Optional[StateProfile] = None, max_focus_fields: int = 24,
                             max_anomaly_fields: int = 8, max_state_fields: int = 512) -> Dict[str, Any]:
    """Build the prompt-side state block.

    Format:
      {
        "all": {"字段名":"当前文本值", ...},
        "focus": ["规则命中的字段名", ...],
        "rare": ["训练分布中罕见的字段名", ...]
      }

    `all` enables OOD detection; `focus`/`rare` are only name lists because the
    values are already present in `all`.
    """
    fields: List[str] = []
    for res in retrieval.values():
        for f in _unique_state_fields_from_rules(res.rules):
            if f not in fields:
                fields.append(f)
    fields = fields[:max_focus_fields]

    anomaly_pairs: List[Tuple[str, float]] = []
    if state_profile is not None:
        anomaly_pairs = state_profile.top_anomalies(record, exclude_fields=fields, max_fields=max_anomaly_fields)
    anomaly_fields = [k for k, _ in anomaly_pairs]
    return {
        "all": _all_state_name_text(record, max_state_fields=max_state_fields),
        "focus": _state_field_names(record, fields),
        "rare": _state_field_names(record, anomaly_fields),
    }


def _clean_keyword_list(values: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return out
    for x in values:
        s = str(x).strip()
        if s and s not in out:
            out.append(s)
    return out


def build_answer(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the exact JSON answer object used for SFT/RL supervision."""
    exp = record.get("expected") or {}
    v_lab = normalize_label(exp.get("vehicle_risk", "safe"))
    i_lab = normalize_label(exp.get("instruction_risk", "safe"))
    v_kws = _clean_keyword_list(exp.get("all_vehicle_risk_keywords", [])) if v_lab != "safe" else []
    i_kws = _clean_keyword_list(exp.get("all_instruction_risk_keywords", [])) if i_lab != "safe" else []
    all_kws: List[str] = []
    for s in v_kws + i_kws:
        if s not in all_kws:
            all_kws.append(s)
    return {
        "vehicle_risk": v_lab,
        "instruction_risk": i_lab,
        "all_vehicle_risk_keywords": v_kws,
        "all_instruction_risk_keywords": i_kws,
        "all_risk_keywords": all_kws,
    }


def build_problem(record: Mapping[str, Any], retrieval: Mapping[str, RetrievalResult],
                  top_rules_per_task: int = 3, max_conditions_per_rule: int = 5,
                  state_profile: Optional[StateProfile] = None, include_rule_phrases: bool = False) -> str:
    vehicle_rules = [_rule_to_understanding(r, record=record, max_conditions=max_conditions_per_rule,
                                            include_phrases=include_rule_phrases)
                     for r in retrieval["vehicle"].rules[:top_rules_per_task]]
    instr_rules = [_rule_to_understanding(r, record=record, max_conditions=max_conditions_per_rule,
                                          include_phrases=include_rule_phrases)
                   for r in retrieval["instruction"].rules[:top_rules_per_task]]
    state_ctx = build_full_state_context(record, retrieval, state_profile=state_profile)
    payload = {
        "query": str(record.get("query", "")),
        "state": state_ctx["all"],
        "focus": state_ctx["focus"],
        "rare": state_ctx["rare"],
        "tree_understanding": {
            "vehicle": vehicle_rules,
            "instruction": instr_rules,
        },
        "required_output": {
            "only_json": True,
            "schema": {
                "vehicle_risk": "safe|medium|high",
                "instruction_risk": "safe|medium|high",
                "all_vehicle_risk_keywords": ["简短的解释一下车辆风险；vehicle_risk=safe 时必须为空数组"],
                "all_instruction_risk_keywords": ["简短的解释一下指令风险；instruction_risk=safe 时必须为空数组"],
                "all_risk_keywords": ["前两类关键词去重后的并集"],
            },
        },
    }
    return (
        "你是车载语音指令安全评估器。请分别判断当前车辆状态 vehicle_risk 与执行驾驶员语音指令的 "
        "instruction_risk。等级只能是 safe、medium、high。\n"
        "输入中的 state 是当前车辆全部状态，用于核查参考规则和发现规则以外的风险；"
        "focus 是供参考的规则中命中的字段名列表，rare 是训练分布中罕见的字段名列表。\n"
        "tree_understanding 是树/森林从训练数据自动归纳出的候选理解，不是最终答案；"
        "当完整状态或 query 与候选理解冲突时，应按照实际状态和 query 判断。\n"
        "最终答案必须且只能是一个 JSON 对象，并严格符合 required_output.schema。\n"
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def build_training_record(record: Mapping[str, Any], retrieval: Mapping[str, RetrievalResult],
                          top_rules_per_task: int = 3, max_conditions_per_rule: int = 5,
                          state_profile: Optional[StateProfile] = None,
                          include_rule_phrases: bool = False) -> Dict[str, str]:
    answer = build_answer(record)
    return {
        "problem": build_problem(record, retrieval, top_rules_per_task=top_rules_per_task,
                                 max_conditions_per_rule=max_conditions_per_rule,
                                 state_profile=state_profile, include_rule_phrases=include_rule_phrases),
        "answer": json.dumps(answer, ensure_ascii=False, separators=(",", ":")),
    }


def build_prompts(records: Sequence[Mapping[str, Any]], bank: DualRiskRuleBank, top_k_vehicle: int = 3,
                  top_k_instruction: int = 3, min_precision: float = 0.0,
                  include_label_sidecar: bool = True, state_mode: str = "focused") -> List[Dict[str, str]]:
    """Build the requested SFT/RL dataset rows.

    The output rows intentionally contain only {"problem": str, "answer": str}.
    `include_label_sidecar` and `state_mode` are accepted for CLI compatibility
    with older versions but are not used in v6.
    """
    out: List[Dict[str, str]] = []
    for r in records:
        retrieval = bank.retrieve(r, top_k_vehicle=top_k_vehicle, top_k_instruction=top_k_instruction,
                                  min_precision=min_precision)
        out.append(build_training_record(r, retrieval, top_rules_per_task=max(top_k_vehicle, top_k_instruction),
                                         state_profile=getattr(bank, "state_profile", None)))
    return out


# ---------------------------- evaluation helpers ----------------------------

def evaluate_rule_votes(records: Sequence[Mapping[str, Any]], bank: DualRiskRuleBank,
                        top_k_vehicle: int = 5, top_k_instruction: int = 5) -> Dict[str, Any]:
    rows = []
    cv = Counter(); ci = Counter(); nv = 0; ni = 0
    for r in records:
        ret = bank.retrieve(r, top_k_vehicle=top_k_vehicle, top_k_instruction=top_k_instruction)
        gt_v = expected_label(r, "vehicle")
        gt_i = expected_label(r, "instruction")
        pv = ret["vehicle"].predicted_label
        pi = ret["instruction"].predicted_label
        nv += int(pv == gt_v)
        ni += int(pi == gt_i)
        cv[(gt_v, pv)] += 1
        ci[(gt_i, pi)] += 1
        rows.append({"id": record_id(r), "vehicle_true": gt_v, "vehicle_pred": pv, "instruction_true": gt_i, "instruction_pred": pi})
    return {
        "n": len(records),
        "vehicle_accuracy_rule_vote": round(nv / max(1, len(records)), 4),
        "instruction_accuracy_rule_vote": round(ni / max(1, len(records)), 4),
        "vehicle_confusion": {f"{a}->{b}": int(c) for (a, b), c in sorted(cv.items())},
        "instruction_confusion": {f"{a}->{b}": int(c) for (a, b), c in sorted(ci.items())},
        "rows": rows,
    }


# ----------------------------------- CLI ------------------------------------

def _add_common_train_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-estimators", type=int, default=80)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--min-samples-leaf", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--max-query-features", type=int, default=1200)
    p.add_argument("--min-query-df", type=int, default=2, help="drop query char n-grams that occur in fewer training cases")
    p.add_argument("--max-auto-categorical-cardinality", type=int, default=20)


def cmd_train(args: argparse.Namespace) -> None:
    records = load_jsonl(args.train, split=args.split)
    if not records:
        raise SystemExit("No training records loaded")
    bank = DualRiskRuleBank.fit(
        records,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        max_query_features=args.max_query_features,
        min_query_df=args.min_query_df,
        max_auto_categorical_cardinality=args.max_auto_categorical_cardinality,
    )
    bank.save(args.model_out)
    if args.rules_out:
        bank.export_rules(args.rules_out)
    summary = {"train": bank.metadata, "self_eval": evaluate_rule_votes(records, bank)}
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_build_prompts(args: argparse.Namespace) -> None:
    bank = DualRiskRuleBank.load(args.model)
    records = load_jsonl(args.input, split=args.split)
    prompts = build_prompts(records, bank, top_k_vehicle=args.top_k_vehicle, top_k_instruction=args.top_k_instruction,
                            min_precision=args.min_precision, include_label_sidecar=not args.no_label_sidecar,
                            state_mode=args.state_mode)
    jsonl_path, parquet_path = write_dataset_outputs(
        args.output, prompts,
        parquet_path=getattr(args, "parquet_output", None),
        skip_parquet=getattr(args, "skip_parquet", False),
    )
    if args.eval_out:
        ev = evaluate_rule_votes(records, bank, top_k_vehicle=args.top_k_vehicle, top_k_instruction=args.top_k_instruction)
        Path(args.eval_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.eval_out).write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n_records": len(prompts), "jsonl": str(jsonl_path), "parquet": str(parquet_path) if parquet_path else None}, ensure_ascii=False, indent=2))


def cmd_oof_prompts(args: argparse.Namespace) -> None:
    records = load_jsonl(args.train, split=args.split)
    if not records:
        raise SystemExit("No training records loaded")
    y_joint = [expected_label(r, "vehicle") + "|" + expected_label(r, "instruction") for r in records]
    n_splits = max(2, args.folds)
    # Some variants may have rare joint labels. StratifiedKFold fails when a
    # class has only one example, so fall back to ordinary KFold in that case.
    counts = Counter(y_joint)
    min_count = min(counts.values()) if counts else 2
    if min_count >= 2:
        n_splits = min(n_splits, min_count)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)
        split_iter = skf.split(np.zeros(len(records)), y_joint)
    else:
        from sklearn.model_selection import KFold
        n_splits = min(n_splits, len(records))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)
        split_iter = kf.split(np.zeros(len(records)))
    all_prompts: List[Dict[str, Any]] = [None] * len(records)  # type: ignore
    fold_summaries = []
    for fold, (train_idx, hold_idx) in enumerate(split_iter):
        fold_train = [records[i] for i in train_idx]
        fold_hold = [records[i] for i in hold_idx]
        bank = DualRiskRuleBank.fit(
            fold_train,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state + fold * 37,
            max_query_features=args.max_query_features,
            min_query_df=args.min_query_df,
            max_auto_categorical_cardinality=args.max_auto_categorical_cardinality,
        )
        prompts = build_prompts(fold_hold, bank, top_k_vehicle=args.top_k_vehicle, top_k_instruction=args.top_k_instruction,
                                min_precision=args.min_precision, include_label_sidecar=not args.no_label_sidecar,
                                state_mode=args.state_mode)
        for pos, p in zip(hold_idx, prompts):
            all_prompts[int(pos)] = p
        ev = evaluate_rule_votes(fold_hold, bank, top_k_vehicle=args.top_k_vehicle, top_k_instruction=args.top_k_instruction)
        fold_summaries.append({k: v for k, v in ev.items() if k != "rows"} | {"fold": fold})
    jsonl_path, parquet_path = write_dataset_outputs(
        args.output, all_prompts,
        parquet_path=getattr(args, "parquet_output", None),
        skip_parquet=getattr(args, "skip_parquet", False),
    )
    # Fit final bank on all training records for test/inference dataset construction.
    final_bank = DualRiskRuleBank.fit(
        records,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        max_query_features=args.max_query_features,
        min_query_df=args.min_query_df,
        max_auto_categorical_cardinality=args.max_auto_categorical_cardinality,
    )
    if args.model_out:
        final_bank.save(args.model_out)
    if args.rules_out:
        final_bank.export_rules(args.rules_out)
    summary = {
        "n_train_prompts": len(all_prompts),
        "folds": n_splits,
        "fold_summaries": fold_summaries,
        "final_self_eval": evaluate_rule_votes(records, final_bank, top_k_vehicle=args.top_k_vehicle, top_k_instruction=args.top_k_instruction),
        "jsonl": str(jsonl_path),
        "parquet": str(parquet_path) if parquet_path else None,
    }
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "final_self_eval"}, ensure_ascii=False, indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    """Convenience command: OOF train dataset + final model + test dataset."""
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Build train OOF prompts and final model.
    oof_args = argparse.Namespace(**vars(args))
    oof_args.output = out / "train_dataset_oof.jsonl"
    oof_args.parquet_output = out / "train_dataset_oof.parquet"
    oof_args.skip_parquet = getattr(args, "skip_parquet", False)
    oof_args.model_out = out / "rule_bank.joblib"
    oof_args.rules_out = out / "rules.json"
    oof_args.summary_out = out / "train_oof_summary.json"
    cmd_oof_prompts(oof_args)
    # Build eval/test prompts if supplied.
    if args.eval:
        bp_args = argparse.Namespace(
            model=out / "rule_bank.joblib",
            input=args.eval,
            split=args.eval_split,
            output=out / "eval_dataset.jsonl",
            parquet_output=out / "eval_dataset.parquet",
            skip_parquet=getattr(args, "skip_parquet", False),
            top_k_vehicle=args.top_k_vehicle,
            top_k_instruction=args.top_k_instruction,
            min_precision=args.min_precision,
            no_label_sidecar=args.no_label_sidecar,
            state_mode=args.state_mode,
            eval_out=out / "eval_rule_vote_metrics.json",
        )
        cmd_build_prompts(bp_args)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Automatic rule mining + problem/answer dataset construction for vehicle instruction risk RL")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("train", help="fit final rule bank")
    sp.add_argument("--train", required=True)
    sp.add_argument("--split", default=None)
    sp.add_argument("--model-out", required=True)
    sp.add_argument("--rules-out", default=None)
    sp.add_argument("--summary-out", default=None)
    _add_common_train_args(sp)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("build-prompts", help="build problem/answer dataset using a fitted rule bank")
    sp.add_argument("--model", required=True)
    sp.add_argument("--input", required=True)
    sp.add_argument("--split", default=None)
    sp.add_argument("--output", required=True, help="JSONL output path; a Parquet sibling is written by default")
    sp.add_argument("--parquet-output", default=None, help="optional explicit Parquet output path")
    sp.add_argument("--skip-parquet", action="store_true", help="debug only: do not write Parquet")
    sp.add_argument("--top-k-vehicle", type=int, default=3)
    sp.add_argument("--top-k-instruction", type=int, default=3)
    sp.add_argument("--min-precision", type=float, default=0.50)
    sp.add_argument("--no-label-sidecar", action="store_true")
    sp.add_argument("--state-mode", choices=["focused", "expanded", "full"], default="focused",
                    help="kept for backward compatibility; v6 always includes full name:text state plus focus/rare field names")
    sp.add_argument("--eval-out", default=None)
    sp.set_defaults(func=cmd_build_prompts)

    sp = sub.add_parser("oof-prompts", help="build out-of-fold problem/answer dataset for training data and fit final rule bank")
    sp.add_argument("--train", required=True)
    sp.add_argument("--split", default=None)
    sp.add_argument("--output", required=True, help="JSONL output path; a Parquet sibling is written by default")
    sp.add_argument("--parquet-output", default=None, help="optional explicit Parquet output path")
    sp.add_argument("--skip-parquet", action="store_true", help="debug only: do not write Parquet")
    sp.add_argument("--model-out", default=None)
    sp.add_argument("--rules-out", default=None)
    sp.add_argument("--summary-out", default=None)
    sp.add_argument("--folds", type=int, default=5)
    sp.add_argument("--top-k-vehicle", type=int, default=3)
    sp.add_argument("--top-k-instruction", type=int, default=3)
    sp.add_argument("--min-precision", type=float, default=0.50)
    sp.add_argument("--no-label-sidecar", action="store_true")
    sp.add_argument("--state-mode", choices=["focused", "expanded", "full"], default="focused",
                    help="kept for backward compatibility; v6 always includes full name:text state plus focus/rare field names")
    _add_common_train_args(sp)
    sp.set_defaults(func=cmd_oof_prompts)

    sp = sub.add_parser("run", help="OOF train dataset + final model + optional eval dataset")
    sp.add_argument("--train", required=True)
    sp.add_argument("--split", default=None, help="split value for the train file, e.g. train")
    sp.add_argument("--eval", default=None, help="optional eval/test JSONL")
    sp.add_argument("--eval-split", default=None)
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--skip-parquet", action="store_true", help="debug only: do not write Parquet outputs")
    sp.add_argument("--folds", type=int, default=5)
    sp.add_argument("--top-k-vehicle", type=int, default=3)
    sp.add_argument("--top-k-instruction", type=int, default=3)
    sp.add_argument("--min-precision", type=float, default=0.50)
    sp.add_argument("--no-label-sidecar", action="store_true")
    sp.add_argument("--state-mode", choices=["focused", "expanded", "full"], default="focused",
                    help="kept for backward compatibility; v6 always includes full name:text state plus focus/rare field names")
    _add_common_train_args(sp)
    sp.set_defaults(func=cmd_run)
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
