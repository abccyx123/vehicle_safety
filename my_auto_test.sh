python3 rule_prompt_miner.py run \
  --train vehicle_instruction_risk_all.jsonl \
  --eval vehicle_instruction_risk_all.jsonl \
  --output-dir outputs/run \
  --folds 5 \
  --top-k-vehicle 3 \
  --top-k-instruction 3 \
  --min-precision 0.50
  # --split train \
  # --eval-split test \