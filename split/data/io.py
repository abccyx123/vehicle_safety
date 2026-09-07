# data/io.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("Writing Parquet requires pandas plus pyarrow/fastparquet") from e
    df = pd.DataFrame(list(records), columns=["problem", "answer"])
    try:
        df.to_parquet(path, index=False, engine="pyarrow")
    except Exception as e:
        raise RuntimeError("Writing Parquet failed. Install pyarrow, e.g. `pip install pyarrow`.") from e

def write_dataset_outputs(jsonl_path: str | Path, records: Sequence[Mapping[str, Any]],
                          parquet_path: Optional[str | Path] = None, skip_parquet: bool = False) -> Tuple[Path, Optional[Path]]:
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