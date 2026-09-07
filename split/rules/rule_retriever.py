# rules/rule_retriever.py
from __future__ import annotations
from typing import Dict, List, Mapping, Tuple
from rules.rule_models import MinedRule, RetrievalResult

def retrieve_rules(miner, record: Mapping[str, Any], top_k: int = 6, min_precision: float = 0.0,
                   prefer_non_safe: bool = True) -> RetrievalResult:
    """
    miner: AutoRuleMiner instance
    """
    if miner.model is None:
        raise RuntimeError("model not fitted")
    X = miner.featurizer.transform_one(record)
    leaf_ids = miner.model.apply(X)[0].tolist()
    votes = miner.predict_proba_label_votes(X)
    predicted_label = max(votes.items(), key=lambda kv: kv[1])[0]
    candidates: List[MinedRule] = []
    for tree_idx, leaf_id in enumerate(leaf_ids):
        rule = miner.rule_by_tree_leaf.get((tree_idx, int(leaf_id)))
        if rule and rule.precision >= min_precision:
            candidates.append(rule)
    # Deduplicate
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
        non_safe_bonus = 1.35 if prefer_non_safe and rule.label != "safe" else 0.62
        lift = rule.precision / max(1e-6, votes.get(rule.label, 1e-6))
        return rule.score_base * pred_bonus * non_safe_bonus * min(2.0, max(0.6, lift))
    non_safe = [r for r in cand if r.label != "safe"]
    safe = [r for r in cand if r.label == "safe"]
    non_safe.sort(key=rank, reverse=True)
    safe.sort(key=rank, reverse=True)
    if non_safe:
        selected = non_safe[:top_k]
        if predicted_label == "safe" and len(selected) < top_k and safe:
            selected.append(safe[0])
    else:
        selected = safe[:min(1, top_k)]
    return RetrievalResult(task=miner.task, predicted_label=predicted_label, label_votes=votes, rules=selected[:top_k])
