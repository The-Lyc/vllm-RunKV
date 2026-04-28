import json
import numpy as np
from collections import defaultdict

def analyze_step_variance(file_path):
    # 1. 按 step 组织数据
    step_groups = defaultdict(list)
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                step = data.get("step")
                imbalance = data.get("imbalance_ms")
                if step is not None and imbalance is not None:
                    step_groups[step].append(imbalance)
            except json.JSONDecodeError:
                continue

    # 2. 计算每个 step 内部的统计量
    step_stds = []
    step_means = []
    
    # 按照 step 编号排序处理
    sorted_steps = sorted(step_groups.keys())
    
    print(f"{'Step':<10} | {'样本数':<8} | {'均值 (ms)':<12} | {'标准差 (σ_step)':<15}")
    print("-" * 55)

    for step in sorted_steps:
        data_points = np.array(step_groups[step])
        if len(data_points) > 1:
            s_std = np.std(data_points, ddof=1)
            s_mean = np.mean(data_points)
            step_stds.append(s_std)
            step_means.append(s_mean)
            if step == 0:
                continue
            # # 打印前 5 个 step 作为样例查看
            # if step < 5:
            print(f"{step:<10} | {len(data_points):<8} | {s_mean:<12.4f} | {s_std:<15.4f}")
        elif step < 5:
            print(f"{step:<10} | {len(data_points):<8} | {'数据不足':<12}")

    print("...")
    
    # 3. 计算“元统计”数据 (Meta-statistics)
    step_stds = np.array(step_stds)
    
    meta_stats = {
        "mean_of_stds": np.mean(step_stds),      # 平均噪声水平
        "std_of_stds": np.std(step_stds, ddof=1), # 噪声的波动程度（异方差性）
        "max_std": np.max(step_stds),            # 最坏情况下的噪声强度
        "min_std": np.min(step_stds),            # 最理想情况下的噪声强度
        "p50_std": np.median(step_stds),         # 中位数噪声
        "p95_std": np.percentile(step_stds, 95)  # 95% 场景下的噪声上限
    }

    return meta_stats

# 执行分析
file_name = '/home/lyc/inference/vllm/exp_results/opt_feedback_observation/opt_component_mfu_1000_20260422_174059.flat.jsonl' # 替换为你的文件名
results = analyze_step_variance(file_name)

print("\n" + "="*40)
print("      所有 Step 噪声特征 (Meta-Stats)      ")
print("="*40)
print(f"平均噪声水平 (Mean σ):       {results['mean_of_stds']:.4f} ms")
print(f"噪声的波动程度 (Std of σs):  {results['std_of_stds']:.4f} ms")
print(f"中位数噪声 (P50 σ):          {results['p50_std']:.4f} ms")
print(f"噪声上限 (P95 σ):            {results['p95_std']:.4f} ms")
print(f"最大/最小噪声差值:           {results['max_std'] - results['min_std']:.4f} ms")
print("="*40)