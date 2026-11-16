#!/usr/bin/env bash
# Problem 2 的訓練排程（一次一個，避免搶 CPU）：
#   1. 等 Problem 3 的排程跑完（p3_runs.txt 裡出現 student）
#   2. baseline：助教的 packaged recipe（不加 --config），跑 1800 iterations
#   3. 老師 1500 iterations（可以看到物體是什麼）
#   4. 把老師複製到 runs/multi_teacher
#   5. 學生 300 iterations（從歷史推測物體；這個是要交的）
# 老師 1500 + 學生 300 = 1800，跟 baseline 用一樣多的資料量（作業要求「same sample budget」）。
# 每個 run 的資料夾會記在 p2_runs.txt。
#
# 用法（在 tmux 裡）：bash run_p2.sh

cd ~/rpl/HW2 || exit 1
export PYTHONUNBUFFERED=1
hw2() { env -u VIRTUAL_ENV uv run --project ../UniLab --no-sync hw2 "$@"; }
latest() { ls -td logs/rsl_rl_ppo/AllegroInhandMulti/*/ | head -1; }

echo "[$(date +%H:%M)] 等待 Problem 3 結束（p3_runs.txt 出現 student、而且沒有 hw2 train 在執行）..."
while ! grep -q "^student:" p3_runs.txt 2>/dev/null || pgrep -f "hw2 train" > /dev/null; do
  sleep 60
done

echo "[$(date +%H:%M)] 開始 baseline（packaged recipe）"
hw2 train multi/yaw_absolute algo.max_iterations=1800 2>&1 | tee p2_baseline.log
echo "baseline: $(latest)" >> p2_runs.txt

echo "[$(date +%H:%M)] 開始老師"
hw2 train multi/yaw_absolute --config multi_teacher.yaml 2>&1 | tee p2_teacher.log
T=$(latest)
echo "teacher: $T" >> p2_runs.txt
mkdir -p runs && rm -rf runs/multi_teacher && cp -r "$T" runs/multi_teacher

echo "[$(date +%H:%M)] 開始學生"
hw2 train multi/yaw_absolute --config multi_student.yaml 2>&1 | tee p2_student.log
echo "student: $(latest)" >> p2_runs.txt

echo "[$(date +%H:%M)] Problem 2 全部訓練完成，run 資料夾記在 p2_runs.txt"
