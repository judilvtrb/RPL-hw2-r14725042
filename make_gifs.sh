#!/usr/bin/env bash
# 為報告錄製所有 GIF：hw2 play 錄 mp4 → ffmpeg 轉成 GIF，存到 media/
cd ~/rpl/HW2 || exit 1
hw2() { env -u VIRTUAL_ENV uv run --project ../UniLab --no-sync hw2 "$@"; }
# 用 UniLab 環境裡已經裝好的 ffmpeg（不用另外安裝）
FF=$(env -u VIRTUAL_ENV uv run --project ../UniLab --no-sync python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
mkdir -p media

clip() {  # clip <GIF 名稱> <cell> <run 資料夾> <config>
  echo "[$(date +%H:%M)] 錄製 $1"
  hw2 play "$2" algo.load_run="$3" training.play_render_mode=record training.cam_distance=2.4 training.cam_elevation=-35 --config "$4" > "media/$1.play.log" 2>&1
  "$FF" -y -loglevel error -i "$3/play_video.mp4" \
    -vf "fps=10,scale=480:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" "media/$1.gif"
  ls -lh "media/$1.gif"
}

clip p1_so3_ball       so3_absolute/ball       runs/so3_ball       my_config.yaml
clip p1_so3_cube       so3_absolute/cube       runs/so3_cube       my_config.yaml
clip p1_so3_cylinder   so3_absolute/cylinder   runs/so3_cyl        my_config.yaml
clip p2_multi_student  multi/yaw_absolute      runs/multi_student  multi_student.yaml
clip p3_adapt_baseline adapt/yaw_absolute/ball runs/adapt_baseline adapt_baseline.yaml
clip adv_bc_ball       so3_absolute/ball       runs/il_bc_ball     il_bc.yaml
echo "[$(date +%H:%M)] 全部完成"
