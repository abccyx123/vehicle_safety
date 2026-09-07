# features/combined_featurizer.py
from __future__ import annotations
from typing import List, Mapping, Optional, Sequence
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from features.state_vectorizer import StateVectorizer

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