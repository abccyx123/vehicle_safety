vehicle_safety/
│
├── README.md
├── requirements.txt
│
├── main.py                          # 统一入口（替代原来的 __main__ 调用）
├── cli.py                           # 命令行参数解析器（可拆分到 main.py 里）
│
├── data/                            # 数据层（IO + 预处理）
│   ├── __init__.py
│   ├── io.py                        # JSONL/Parquet 读写（load_jsonl, write_jsonl, write_parquet）
│   ├── state_parser.py              # 状态字段解析（ensure_state, infer_raw_to_readable, _to_float, _norm_cat）
│   └── label_utils.py               # 标签/关键词规范化（normalize_label, max_risk, task_phrases, _clean_keyword_list）
│
├── features/                        # 特征工程层（【数据层】的全部）
│   ├── __init__.py
│   ├── state_vectorizer.py          # StateVectorizer（含 FieldProfile, StateFeatureSpec）
│   ├── state_profile.py             # StateProfile（MAD/分位数异常检测，独立于特征向量化）
│   └── combined_featurizer.py       # CombinedFeaturizer（state + query N-gram）
│
├── rules/                           # 规则挖掘层（【工序1】的全部）
│   ├── __init__.py
│   ├── rule_models.py               # 数据类：RuleCondition, MinedRule, RetrievalResult
│   ├── rule_miner.py                # AutoRuleMiner（训练森林 + 解剖叶子 + B1基础分计算）
│   ├── rule_retriever.py            # 检索逻辑（B2修正分 + rank排序 + Top-K筛选）
│   └── rule_bank.py                 # DualRiskRuleBank（组合车辆/指令矿工 + 持久化）
│
├── prompts/                         # Prompt构建层（【工序2】的全部）
│   ├── __init__.py
│   ├── state_context.py             # build_full_state_context（all/focus/rare）
│   ├── rule_translator.py           # _rule_evidence, _rule_to_understanding, _display_state_name_value
│   ├── problem_builder.py           # build_problem（组装最终Prompt）
│   ├── answer_builder.py            # build_answer（构建监督JSON答案）
│   └── dataset_builder.py           # build_training_record, build_prompts（串联生成数据集）
│
├── evaluation/                      # 评估层
│   ├── __init__.py
│   └── rule_evaluator.py            # evaluate_rule_votes（随机森林自身预测准确率）
│
├── pipelines/                       # 流水线调度层（跨模块组合）
│   ├── __init__.py
│   ├── train_pipeline.py            # cmd_train 的工程化版本
│   ├── build_prompts_pipeline.py    # cmd_build_prompts 的工程化版本
│   ├── oof_prompts_pipeline.py      # cmd_oof_prompts 的工程化版本（含5折交叉验证）
│   └── run_pipeline.py              # cmd_run 的工程化版本（全流程一键跑通）
│
└── utils/                           # 跨模块共享工具
    ├── __init__.py
    ├── constants.py                 # RISK_ORDER, SEVERITY_WEIGHT, 默认超参数
    └── logger.py                    # 统一日志输出（可后续扩展）