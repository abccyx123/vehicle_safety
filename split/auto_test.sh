PYTHONPATH=. python3 main.py run \
  --train vehicle_instruction_risk_all.jsonl \
  --eval vehicle_instruction_risk_all.jsonl \
  --output-dir outputs/test_reconstruct \
  --skip-parquet
  # --folds 5 \
  # --top-k-vehicle 3 \
  # --top-k-instruction 3 \
  # --min-precision 0.50 \
#有默认值尽量不用再传参