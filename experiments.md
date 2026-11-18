# 實驗紀錄

## 計分 cell 總覽

| cell | 交出去的 run | eval 分數 | tier |
|---|---|---|---|
| so3_absolute/ball | runs/so3_ball | 8.00 | 3 |
| so3_absolute/cube | runs/so3_cube | 3.36 | 3 |
| so3_absolute/cylinder | runs/so3_cyl | 4.50 | 3 |
| multi/yaw_absolute | runs/multi_student | 5.52（最差物體） | 3 |
| adapt/yaw_absolute/ball | 待決定（baseline 23.73 / RMA 學生 19.55） | | 3 |

## 所有實驗

| 日期 | cell | run 資料夾 | 改了什麼 | iterations | 訓練最後的 goals | eval 分數 | 觀察 |
|---|---|---|---|---|---|---|---|
| 11/16 | yaw_absolute/ball | 2026-11-16_05-40-49_mujoco | baseline（my_config.yaml） | 400 | ~3.0 | 3.27 (tier 3) | 流程測試；action std 從 1.00 升到 1.24 |
| 11/16 | so3_absolute/ball | 2026-11-16_11-52-16_mujoco | baseline（my_config.yaml） | 1500 | 6.66 | **8.00 (tier 3)** | 訓練數字比 eval 低，因為訓練時動作有隨機雜訊、eval 是確定性的 |
| 11/16 | so3_absolute/cube | 2026-11-16_12-51-34_mujoco | baseline（my_config.yaml） | 3000 | 3.22 | **3.36 (tier 3)** | 第 1090 次：episode length 385、goals 0.01 → 卡在「抓著不動」的局部最佳解；第 1724 次：goals 1.19、episode length 350 → 自己跳出來了（開始轉之後比較容易掉，所以 episode length 下降） |
| 11/16 | so3_absolute/cylinder | 2026-11-16_14-50-15_mujoco | baseline（my_config.yaml） | 3000 | 4.10 | **4.50 (tier 3)** | |
| 11/16 | adapt/yaw_absolute/ball | 2026-11-16_17-23-25_mujoco | P3 baseline：MLP + domain randomization，critic 可看物理參數（adapt_baseline.yaml） | 3000 | 23.8 | **23.73 (tier 3)** | public：in_range 28.05 / light 29.17 / heavy 19.95 / slippery 27.67 / sticky 26.55 / off_center 26.80 / weak_motor 20.64 / strong_motor 32.50；heavy 掉落率最高 0.28 |
| 11/16 | adapt/yaw_absolute/ball | 2026-11-16_19-13-42_mujoco | P3 RMA 老師：actor 讀 6 個真實物理參數（adapt_teacher.yaml，--allow-privileged） | 3000 | 25.0 | 21.97（參考用，不能交） | public：in_range 28.06 / light 30.48 / heavy 22.36 / slippery 25.05 / sticky 27.36 / off_center 24.84 / weak_motor 20.30 / strong_motor 33.44；知道物理參數也沒有贏過 baseline |
| 11/16 | adapt/yaw_absolute/ball | 2026-11-16_21-16-25_mujoco | P3 RMA 學生：φ 從 30 步歷史推測 z（adapt_student.yaml） | 500 | 28.3 | **19.55 (tier 3)** | adapt_mse 0.053；public：in_range 27.50 / light 30.56 / heavy 20.84 / slippery 24.83 / sticky 28.89 / off_center 21.98 / weak_motor 20.08 / strong_motor 32.75；slippery 掉落率 0.17、off_center 0.19（baseline 只有 0.03、0.06）→ 超出訓練範圍時 φ 的推測可能失準 |
| 11/16 | multi/yaw_absolute | 2026-11-16_21-35-47_mujoco | P2 baseline：packaged recipe（不加 --config） | 1800 | 9.39（三物體平均） | **4.67 (tier 3)** | 各物體：ball 17.52 / cube 4.67 / cylinder 8.59（平均 10.26） |
| 11/16 | multi/yaw_absolute | 2026-11-16_23-02-26_mujoco | P2 老師：actor 讀物體 one-hot + 物體姿態（multi_teacher.yaml） | 1500 | 約 5.6 | 不能評分 | |
| 11/17 | multi/yaw_absolute | 2026-11-17_00-27-36_mujoco | P2 學生：φ 從 30 步歷史推測物體（multi_student.yaml） | 老師 1500 + 學生 300 = 1800 | | **5.52 (tier 3)** | 各物體：ball 13.33 / cube 5.52 / cylinder 7.97（平均 8.94）；同樣資料量下最差物體 +18%，但 ball −4.2、平均 −1.3 → 拉高最弱的、犧牲最強的 |

## 學到的事

- 同時跑 3 個訓練會慢 10 倍以上（每次 iteration 從 2.2 秒變成 27 秒），原因是 UniLab 把每個 run 綁在所有 CPU 核心上 → 改成排隊依序跑（run_p3.sh、run_p2.sh）
- 訓練時的 goals_reached 不等於正式分數，一定要跑 hw2 eval（訓練有隨機雜訊、而且是在 reset 時平均）
- cube 和 cylinder 會先卡在「抓著不動」好幾百次，需要給足夠的 iterations（3000）才會自己跳出來
- P3：RMA 學生在超出訓練範圍時反而比 baseline 差。推測原因：φ 推測連續的物理數值時，遇到沒看過的範圍會猜錯，base policy 收到錯誤的 ẑ；baseline 學的是一套不依賴推測的穩健策略
- P2：同樣的老師／學生方法有效，因為要推測的只是「三種物體中的哪一種」，而且評分時不會超出範圍
- P2 跟 P3 的對比：特權資訊是「離散、評分時不會超出範圍」的時候，老師／學生有幫助；是「連續、評分時會外推」的時候，反而可能有害