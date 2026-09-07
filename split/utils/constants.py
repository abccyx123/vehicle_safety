# utils/constants.py
RISK_ORDER = {"safe": 0, "medium": 1, "high": 2}
ORDER_RISK = {v: k for k, v in RISK_ORDER.items()}
SEVERITY_WEIGHT = {"safe": 0.82, "medium": 1.08, "high": 1.18}