# Auto Rule Prompt Miner

Automatic rule mining + signal localizer + candidate-hypothesis generator for vehicle-instruction risk assessment. Given a JSONL of `(query, state, expected risk labels)`, it fits a tree/forest rule bank and emits a `{problem, answer}` dataset for downstream training. Outputs are written as both JSONL and Parquet.

`problem` is a single user-side prompt string containing the task description, the input payload, and the required output schema. `answer` is a compact JSON string containing only `vehicle_risk`, `instruction_risk`, `all_vehicle_risk_keywords`, `all_instruction_risk_keywords`, and `all_risk_keywords`. When a risk class is `safe`, its keyword array is forced to be empty.

## Install

```bash
pip install -r requirements.txt
```

## How to run

End-to-end (out-of-fold training set + final rule bank + optional eval set):

```bash
python rule_prompt_miner.py run \
  --train vehicle_instruction_risk_all.jsonl \
  --split train \
  --eval vehicle_instruction_risk_all.jsonl \
  --eval-split test \
  --output-dir outputs/run \
  --folds 5 \
  --top-k-vehicle 3 \
  --top-k-instruction 3 \
  --min-precision 0.50
```

Default outputs:

```text
outputs/run/train_dataset_oof.jsonl
outputs/run/train_dataset_oof.parquet
outputs/run/eval_dataset.jsonl
outputs/run/eval_dataset.parquet
outputs/run/rule_bank.joblib
outputs/run/rules.json
outputs/run/train_oof_summary.json
outputs/run/eval_rule_vote_metrics.json
```

`--skip-parquet` is for debugging only; production datasets write JSONL and Parquet together.

Subcommands `train`, `build-prompts`, and `oof-prompts` are also available for finer control; see `python rule_prompt_miner.py <cmd> --help`.

## Prompt example

The input payload embedded inside `problem` looks like this:

```json
{
  "query": "车速调到130迈",
  "state": {
    "下雨状态": "moderate_rain",
    "车速": "125",
    "纵向加速度": "-0.0224"
  },
  "focus": ["车速", "纵向加速度"],
  "rare": ["下雨状态"],
  "tree_understanding": {
    "vehicle": [
      {
        "tree_understanding": "这些命中证据在训练树叶中主要对应 vehicle_risk=high；这只是候选假设，可被当前完整状态和指令推翻。",
        "risk_hypothesis": "high",
        "evidence": ["车速=125"]
      }
    ],
    "instruction": []
  },
  "required_output": {
    "only_json": true,
    "schema": {
      "vehicle_risk": "safe|medium|high",
      "instruction_risk": "safe|medium|high",
      "all_vehicle_risk_keywords": ["仅车辆状态风险关键词；vehicle_risk=safe 时必须为空数组"],
      "all_instruction_risk_keywords": ["仅执行指令风险关键词；instruction_risk=safe 时必须为空数组"],
      "all_risk_keywords": ["前两类关键词去重后的并集"]
    }
  }
}
```
