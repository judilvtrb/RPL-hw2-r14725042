#!/usr/bin/env bash
# Advanced Challenge 的排程：等 run_p3b.sh 跑完，再對三種物體各跑：
#   - BC（300 iterations）
#   - DAgger（300 iterations）
#   - PPO 從零開始、同樣 300 iterations（對照組：一樣多的資料量，PPO 能學到多少）
# 每個結果都用 seed 0、1、2 評估，存在 eval_results/il_*.json。
# 專家（PPO，1500 或 3000 iterations）的分數也一起用三個 seed 重新評估。
#
# 用法（在 tmux 裡）：bash run_il.sh

cd ~/rpl/HW2 || exit 1
export PYTHONUNBUFFERED=1
hw2() { env -u VIRTUAL_ENV uv run --project ../UniLab --no-sync hw2 "$@"; }
latest() { ls -td logs/rsl_rl_ppo/AllegroInhandReorient/*/ | head -1; }
mkdir -p eval_results runs

echo "[$(date +%H:%M)] 等待 run_p3b.sh 結束..."
while [ ! -f eval_results/p3_histmlp_seed2.json ] || pgrep -f "hw2 (train|eval)" > /dev/null; do
  sleep 60
done

eval3() {  # eval3 <名稱> <cell> <run> <config>
  for seed in 0 1 2; do
    hw2 eval "$2" "$3" --config "$4" --seed $seed --out "eval_results/$1_seed$seed.json" > /dev/null 2>&1
  done
}

for obj in ball cube cyl; do
  case $obj in
    ball) cell=so3_absolute/ball ;;
    cube) cell=so3_absolute/cube ;;
    cyl)  cell=so3_absolute/cylinder ;;
  esac

  echo "[$(date +%H:%M)] $obj：評估 PPO 專家"
  eval3 "il_expert_$obj" "$cell" "runs/so3_$obj" my_config.yaml

  for m in bc dagger; do
    echo "[$(date +%H:%M)] $obj：$m"
    hw2 train "$cell" algo.algorithm.expert_checkpoint="runs/so3_$obj" --config "il_$m.yaml" 2>&1 | tee "il_${m}_$obj.log"
    R=$(latest)
    echo "${m}_$obj: $R" >> il_runs.txt
    mkdir -p "runs/il_${m}_$obj" && cp "$R"/model_299.pt "$R"/run_config.json "runs/il_${m}_$obj/"
    eval3 "il_${m}_$obj" "$cell" "runs/il_${m}_$obj" "il_$m.yaml"
  done

  echo "[$(date +%H:%M)] $obj：PPO 從零開始 300 iterations（對照組）"
  hw2 train "$cell" algo.max_iterations=300 --config my_config.yaml 2>&1 | tee "il_ppo300_$obj.log"
  R=$(latest)
  echo "ppo300_$obj: $R" >> il_runs.txt
  mkdir -p "runs/il_ppo300_$obj" && cp "$R"/model_299.pt "$R"/run_config.json "runs/il_ppo300_$obj/"
  eval3 "il_ppo300_$obj" "$cell" "runs/il_ppo300_$obj" my_config.yaml
done

echo "[$(date +%H:%M)] Advanced 全部完成，run 資料夾記在 il_runs.txt"
