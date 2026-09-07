# cli.py
from __future__ import annotations
import argparse

from pipelines.build_prompts_pipeline import run_build_prompts
from pipelines.oof_prompts_pipeline import run_oof_prompts
from pipelines.run_pipeline import run_pipeline
from pipelines.train_pipeline import run_train


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Automatic rule mining + problem/answer dataset construction for vehicle instruction risk RL")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("train", help="fit final rule bank")
    sp.add_argument("--train", required=True)
    sp.add_argument("--split", default=None)
    sp.add_argument("--model-out", required=True)
    sp.add_argument("--rules-out", default=None)
    sp.add_argument("--summary-out", default=None)
    _add_common_args(sp)
    sp.set_defaults(func=lambda args: run_train(
        args.train,
        args.model_out,
        split=args.split,
        rules_out=args.rules_out,
        summary_out=args.summary_out,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        max_query_features=args.max_query_features,
        min_query_df=args.min_query_df,
        max_auto_categorical_cardinality=args.max_auto_categorical_cardinality,
    ))

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
    sp.set_defaults(func=lambda args: run_build_prompts(
        args.model,
        args.input,
        args.output,
        split=args.split,
        parquet_output=args.parquet_output,
        skip_parquet=args.skip_parquet,
        top_k_vehicle=args.top_k_vehicle,
        top_k_instruction=args.top_k_instruction,
        min_precision=args.min_precision,
        no_label_sidecar=args.no_label_sidecar,
        state_mode=args.state_mode,
        eval_out=args.eval_out,
    ))

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
    _add_common_args(sp)
    sp.set_defaults(func=lambda args: run_oof_prompts(
        args.train,
        args.output,
        split=args.split,
        parquet_output=args.parquet_output,
        skip_parquet=args.skip_parquet,
        model_out=args.model_out,
        rules_out=args.rules_out,
        summary_out=args.summary_out,
        folds=args.folds,
        top_k_vehicle=args.top_k_vehicle,
        top_k_instruction=args.top_k_instruction,
        min_precision=args.min_precision,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        max_query_features=args.max_query_features,
        min_query_df=args.min_query_df,
        max_auto_categorical_cardinality=args.max_auto_categorical_cardinality,
        no_label_sidecar=args.no_label_sidecar,
        state_mode=args.state_mode,
    ))

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
    _add_common_args(sp)
    sp.set_defaults(func=lambda args: run_pipeline(
        args.train,
        args.output_dir,
        eval_path=args.eval,
        split=args.split,
        eval_split=args.eval_split,
        skip_parquet=args.skip_parquet,
        folds=args.folds,
        top_k_vehicle=args.top_k_vehicle,
        top_k_instruction=args.top_k_instruction,
        min_precision=args.min_precision,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        max_query_features=args.max_query_features,
        min_query_df=args.min_query_df,
        max_auto_categorical_cardinality=args.max_auto_categorical_cardinality,
        no_label_sidecar=args.no_label_sidecar,
        state_mode=args.state_mode,
    ))
    return p


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-estimators", type=int, default=80)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--min-samples-leaf", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--max-query-features", type=int, default=1200)
    p.add_argument("--min-query-df", type=int, default=2,
                   help="drop query char n-grams that occur in fewer training cases")
    p.add_argument("--max-auto-categorical-cardinality", type=int, default=20)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
