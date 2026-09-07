# PYTHONPATH=. python3 main.py train \
#     --train ../vehicle_instruction_risk_all.jsonl \
#     --model-out outputs/rule_bank.joblib \
#     --rules-out outputs/rules.json \
#     --summary-out outputs/summary.json \
#     --n-estimators 5 \
#     --max-depth 6 \
#     --min-samples-leaf 1 \
#     --random-state 42


PYTHONPATH=. python3 main.py run \
  --train vehicle_instruction_risk_all.jsonl \
  --eval vehicle_instruction_risk_all.jsonl \
  --output-dir outputs/test_reconstruct \
  --folds 5 \
  --top-k-vehicle 3 \
  --top-k-instruction 3 \
  --min-precision 0.50 \
  --skip-parquet