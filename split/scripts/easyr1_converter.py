# scripts/easyr1_converter.py
'''运行方法：需要两个参数，--input 和 --output。
--input 是 split 项目产出的 jsonl 文件，--output 是 EasyR1 格式的 jsonl 文件。
--input 和 --output 都是必填参数，不能省略。
/split下执行
python3 scripts/easyr1_converter.py \
  --input outputs/test_reconstruct/train_dataset_oof.jsonl \
  --output convertOutputs/train_easyr1.jsonl
python3 scripts/easyr1_converter.py \
  --input outputs/test_reconstruct/eval_dataset.jsonl \
  --output convertOutputs/eval_easyr1.jsonl
'''
"""
将 split/ 项目产出的 {"problem", "answer"} 格式，
转换为 EasyR1 训练所需的 {"prompt", "answer", "data_source"} 格式。
"""
import json
import argparse
from pathlib import Path


def convert_split_to_easyr1(
    input_path: str,
    output_path: str,
    data_source: str = "vehicle_safety",
) -> None:
    """逐行转换，保留所有原始信息，只做字段重命名和字段补充。"""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_lines = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # 核心转换：problem -> prompt
            new_r = {
                "prompt": r["problem"],
                "answer": r["answer"],
                "data_source": data_source,  # EasyR1 用于区分数据源
            }
            # 保留原始字段（可选），方便调试
            # new_r["_original_problem"] = r["problem"]
            fout.write(json.dumps(new_r, ensure_ascii=False, separators=(",", ":")) + "\n")
            n_lines += 1
    print(f"Converted {n_lines} lines: {input_path} -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="split 项目产出的 jsonl")
    parser.add_argument("--output", required=True, help="EasyR1 格式的 jsonl")
    parser.add_argument("--data-source", default="vehicle_safety")
    args = parser.parse_args()
    convert_split_to_easyr1(args.input, args.output, args.data_source)


if __name__ == "__main__":
    main()