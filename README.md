

# Pose3DM



### Environment

The project is developed under the following environment:

- Python 3.10.x
- PyTorch 2.2.1
- CUDA 12.1

1. `pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/`
2. `cd kernels/selective_scan && pip install -e .`


### Dataset

Preprocessing

1. Download the fine-tuned Stacked Hourglass detections of [MotionBERT](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md)'s preprocessed H3.6M data [here](https://onedrive.live.com/?authkey=%21AMG5RlzJp%2D7yTNw&id=A5438CD242871DF0%21206&cid=A5438CD242871DF0) and unzip it to 'data/motion3d', or direct download our processed data here and unzip it.
2. Slice the motion clips by running the following python code in `tools/convert_h36m.py`

`python convert_h36m.py`

### Training

After preparing the dataset, you can train the model using the following steps:

You can train Human3.6M with the following command:
`python train.py --config <PATH-TO-CONFIG> --checkpoint <PATH-TO-CHECKPOINT>`

For example:

`python train.py --config configs/pose3d/Pose3DM_train_h36m_B.yaml --checkpoint checkpoint/pose3d/MB_train_h36m`


### Evaluation
We provide [Pose3DM_B](https://drive.google.com/file/d/123AA9GDnnnbkiGuK-VoynY4bx4wPIn_1/view?usp=drive_link). You can download and get pretrained weight.

`python train.py --eval-only --checkpoint <CHECKPOINT-DIRECTORY> --checkpoint-file <CHECKPOINT-FILE-NAME> --config <PATH-TO-CONFIG>`

For example:

`python train.py --config checkpoint/pose3d/Pose3DM_B/config.yaml --evaluate checkpoint/pose3d/Pose3DM_B/best_epoch.bin --checkpoint eval/checkpoint`

### Demo

Our demo is based on a modified version of the [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer) repository. To begin, download the [YOLOv3](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA) and [HRNet](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA) pretrained models from the provided link and place them in the './demo/lib/checkpoint' directory. Next, download our [Pose3DM_B](https://drive.google.com/file/d/123AA9GDnnnbkiGuK-VoynY4bx4wPIn_1/view?usp=drive_link) checkpoint from the specified link and store it in the './checkpoint' directory. After that, place your in-the-wild videos in the './demo/video' directory. Run the command below:

`python vis.py --video AIG.mp4 --gpu 0`



### Copyright Notice

This project is licensed under the MIT License. For more details, please refer to the [LICENSE.txt](https://github.com/Reus3237/Pose3DM/blob/Pose3DM/LICENSE.txt)

### Acknowledgement

Our code refers to the following repositories:

- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer)
- [PoseMamba](https://github.com/nankingjing/PoseMamba)






