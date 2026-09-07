# rules/rule_miner.py
from __future__ import annotations
import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from data.label_utils import expected_label, normalize_label, task_phrases
from data.io import record_id
from features.combined_featurizer import CombinedFeaturizer
from rules.rule_models import MinedRule, RuleCondition
from utils.constants import SEVERITY_WEIGHT

def _feature_condition_text(feature_name: str, op: str, threshold: float):
    positive = op == ">"
    if feature_name.startswith("q:"):
        frag = feature_name[2:]
        if threshold <= 0.5:
            if op == ">":
                return f"query 包含『{frag}』", None, frag, True
            return f"query 不包含『{frag}』", None, frag, False
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
                    if expected_label(records[i], self.task) == label:
                        for p in task_phrases(records[i], self.task):
                            phrase_counter[p] += 1
                phrases = [p for p, _ in phrase_counter.most_common(5)]
                exemplar_ids = [record_id(records[i]) for i in idxs[:3]]
                conds = paths.get(int(leaf_id), [])
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
                "conditions": [{"feature_index": c.feature_index, "feature_name": c.feature_name, "op": c.op,
                                "threshold": c.threshold, "text": c.text, "raw_field": c.raw_field,
                                "query_fragment": c.query_fragment, "positive": c.positive}
                               for c in r.conditions[:max_conditions]],
                "phrases": r.phrases,
                "exemplar_ids": r.exemplar_ids,
                "score_base": round(r.score_base, 6),
            })
        return out

    def retrieve(self, record: Mapping[str, Any], top_k: int = 6, min_precision: float = 0.0, #??新增
            prefer_non_safe: bool = True) -> RetrievalResult:
        from rules.rule_retriever import retrieve_rules
        return retrieve_rules(self, record, top_k, min_precision, prefer_non_safe)
    