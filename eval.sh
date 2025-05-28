CUDA_VISIBLE_DEVICES=2 python train.py \
  --config checkpoint/pose3d/Pose3DM_B/config.yaml \
  --evaluate checkpoint/pose3d/Pose3DM_B/best_epoch.bin \
  --checkpoint eval/checkpoint
# CUDA_VISIBLE_DEVICES=2 python train.py \
#   --config checkpoint/pose3d/Pose3DM_S/config.yaml \
#   --evaluate checkpoint/pose3d/Pose3DM_S/best_epoch.bin \
#   --checkpoint eval/checkpoint
# CUDA_VISIBLE_DEVICES=2 python train.py \
#   --config checkpoint/pose3d/Pose3DM_L/config.yaml \
#   --evaluate checkpoint/pose3d/Pose3DM_L/best_epoch.bin \
#   --checkpoint eval/checkpoint