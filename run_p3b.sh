#!/usr/bin/env bash
# P3 第二輪：1) 換種子重新評估  2) 訓練長歷史 MLP  3) 評估長歷史 MLP
cd ~/rpl/HW2 || exit 1
export PYTHONUNBUFFERED=1
hw2() { env -u VIRTUAL_ENV uv run --project ../UniLab --no-sync hw2 "$@"; }
mkdir -p eval_results

echo "[$(date +%H:%M)] 1) 用 seed 1、2 重新評估 baseline 和 RMA 學生"
for seed in 1 2; do
  hw2 eval adapt/yaw_absolute/ball runs/adapt_baseline --config adapt_baseline.yaml --seed $seed --out eval_results/p3_baseline_seed$seed.json > /dev/null 2>&1
  hw2 eval adapt/yaw_absolute/ball runs/adapt_student --config adapt_student.yaml --seed $seed --out eval_results/p3_student_seed$seed.json > /dev/null 2>&1
done

echo "[$(date +%H:%M)] 2) 訓練長歷史 MLP"
hw2 train adapt/yaw_absolute/ball --config adapt_histmlp.yaml 2>&1 | tee p3_histmlp.log
R=$(ls -td logs/rsl_rl_ppo/AllegroInhandReorient/*/ | head -1)
echo "histmlp: $R" >> p3_runs.txt
mkdir -p runs/adapt_histmlp && cp "$R"/model_2999.pt "$R"/run_config.json runs/adapt_histmlp/

echo "[$(date +%H:%M)] 3) 評估長歷史 MLP"
hw2 eval adapt/yaw_absolute/ball runs/adapt_histmlp --config adapt_histmlp.yaml --challenge public > eval_results/p3_histmlp.txt 2>&1
for seed in 1 2; do
  hw2 eval adapt/yaw_absolute/ball runs/adapt_histmlp --config adapt_histmlp.yaml --seed $seed --out eval_results/p3_histmlp_seed$seed.json > /dev/null 2>&1
done

echo "[$(date +%H:%M)] 全部完成"