#!/usr/bin/env bash
# Problem 3 的訓練排程（一次一個，避免搶 CPU）：
#   1. 等 Problem 1 的訓練全部結束
#   2. baseline（一般 MLP，不做 adaptation，當對照組）
#   3. RMA 老師（第一階段，可以看到真實物理參數）
#   4. 把老師複製到 runs/adapt_teacher
#   5. RMA 學生（第二階段，學會從歷史推測物理參數；這個是要交的）
# 每個 run 的資料夾會記在 p3_runs.txt。
#
# 用法（在 tmux 裡）：bash run_p3.sh

cd ~/rpl/HW2 || exit 1
export PYTHONUNBUFFERED=1
# alias 在 script 裡不能用，所以用 function 定義一樣的 hw2 指令
hw2() { env -u VIRTUAL_ENV uv run --project ../UniLab --no-sync hw2 "$@"; }
latest() { ls -td logs/rsl_rl_ppo/AllegroInhandReorient/*/ | head -1; }

echo "[$(date +%H:%M)] 等待 Problem 1 結束（cylinder 跑完、而且沒有任何 hw2 train 在執行）..."
while ! grep -q "Learning iteration 2999/3000" so3_cyl.log 2>/dev/null || pgrep -f "hw2 train" > /dev/null; do
  sleep 60
done

echo "[$(date +%H:%M)] 開始 baseline"
hw2 train adapt/yaw_absolute/ball --config adapt_baseline.yaml 2>&1 | tee p3_baseline.log
echo "baseline: $(latest)" >> p3_runs.txt

echo "[$(date +%H:%M)] 開始 RMA 老師"
hw2 train adapt/yaw_absolute/ball --config adapt_teacher.yaml 2>&1 | tee p3_teacher.log
T=$(latest)
echo "teacher: $T" >> p3_runs.txt
mkdir -p runs && rm -rf runs/adapt_teacher && cp -r "$T" runs/adapt_teacher

echo "[$(date +%H:%M)] 開始 RMA 學生"
hw2 train adapt/yaw_absolute/ball --config adapt_student.yaml 2>&1 | tee p3_student.log
echo "student: $(latest)" >> p3_runs.txt

echo "[$(date +%H:%M)] Problem 3 全部訓練完成，run 資料夾記在 p3_runs.txt"
