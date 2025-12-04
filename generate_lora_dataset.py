
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def generate_dualtask_dataset(csv_path, output_dir, test_size=0.1, random_seed=42):
    df = pd.read_csv(csv_path)
    df = df.dropna(how="all").reset_index(drop=True)

    completion_samples = []
    classification_samples = []

    for idx, row in df.iterrows():
        fields = row.dropna().to_dict()
        if len(fields) < 2:
            continue

        # 补全任务
        mask_field = np.random.choice(list(fields.keys()))
        target_value = fields.pop(mask_field)

        completion_input = "以下是部分样本：\n" + json.dumps(fields, ensure_ascii=False)
        completion_output = str(target_value)

        completion_samples.append({
            "instruction": f"根据已知字段补全 {mask_field} 的值",
            "input": completion_input,
            "output": completion_output
        })

        # 合格性判断任务
        if "data_min" in row and not pd.isnull(row["data_min"]):
            label = "不合格" if row["data_min"] < 0.1 else "合格"
            classification_input = "以下是完整仿真数据：\n" + json.dumps(row.dropna().to_dict(), ensure_ascii=False)
            classification_output = label

            classification_samples.append({
                "instruction": "根据仿真数据判断产品是否合格",
                "input": classification_input,
                "output": classification_output
            })

    # 混合并划分
    all_samples = completion_samples + classification_samples
    np.random.shuffle(all_samples)
    train, val = train_test_split(all_samples, test_size=test_size, random_state=random_seed)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_jsonl(data, path):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    save_jsonl(train, out_dir / "train.jsonl")
    save_jsonl(val, out_dir / "dev.jsonl")
    print(f"✅ 完成：生成 {len(train)} 条训练样本，{len(val)} 条验证样本。\n输出目录：{out_dir.resolve()}")

# 示例用法
# generate_dualtask_dataset("sim_features.csv", "./llamafactory_dataset")
