video=/workspace/face/data/my_golden_wheel/exp8.mp4
sound=/workspace/face/code/avatarify-python/video/origin_golden_wheel_3p_0_soundtrack.mp3
save=/workspace/face/data/my_golden_wheel/exp8_combine.mp4


ffmpeg -i $video \
-i  $sound \
-c:v copy -c:a aac -strict experimental $save