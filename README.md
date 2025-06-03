


# Bidirectional Mamba-enhanced 3D Human Pose Estimation for Accurate Clinical Gait Analysis

**Published in:** The Visual Computer

## Overview

This project implements a Bidirectional Mamba architecture for monocular 3D human pose estimation, specifically designed for clinical gait analysis. Our approach enhances the accuracy of pose estimation and facilitates gait analysis in clinical settings.

## Environment

The project is developed under the following environment:

- **Python**: 3.10.x
- **PyTorch**: 2.2.1
- **CUDA**: 12.1 (ensure CUDA is installed and configured correctly)

### Setup Instructions

To set up your environment, follow these steps:

1. **Create a new conda environment:**
   ```bash
   conda create -n pose3dm python=3.10.16
   ```
2. **Activate the conda environment:**
   ```bash
   conda activate pose3dm
   ```
3. **Install required packages:**
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
   ```
4. **Install additional Python dependencies:**
   ```bash
   pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
   ```
5. **Install the selective scan kernel:**
   ```bash
   cd kernels/selective_scan && pip install -e .

## Dataset Preparation

### Preprocessing

1. **Download the fine-tuned Stacked Hourglass detections** from [MotionBERT's documentation](https://github.com/Walter0807/MotionBERT/blob/main/docs/pose3d.md) and extract it to `data/motion3d`.

2. **Alternatively, download our processed data** [here](<Insert_download_link>) and unzip it to the same directory.

3. **Slice the motion clips** by executing the following command:
   ```bash
   python tools/convert_h36m.py
   ```

## Training the Model

After preparing the dataset, you can train the model with the following command:

```bash
python train.py --config <PATH-TO-CONFIG> --checkpoint <PATH-TO-CHECKPOINT>
```

**Example command:**
```bash
python train.py --config configs/pose3d/Pose3DM_train_h36m_B.yaml --checkpoint checkpoint/pose3d/MB_train_h36m
```

## Model Evaluation

To evaluate the model, download the pretrained weights for [Pose3DM_B](https://drive.google.com/file/d/123AA9GDnnnbkiGuK-VoynY4bx4wPIn_1/view?usp=drive_link) and place them in the appropriate directory.

Execute the following command to evaluate the model:

```bash
python train.py --eval-only --checkpoint <CHECKPOINT-DIRECTORY> --checkpoint-file <CHECKPOINT-FILE-NAME> --config <PATH-TO-CONFIG>
```

**Example command:**
```bash
python train.py --config checkpoint/pose3d/Pose3DM_B/config.yaml --evaluate checkpoint/pose3d/Pose3DM_B/best_epoch.bin --checkpoint eval/checkpoint
```

## Demo

To demonstrate the model's capabilities, follow these steps:

1. **Download the pretrained models for YOLOv3 and HRNet** from these [links](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA) and place them in the `./demo/lib/checkpoint` directory.

2. **Download our checkpoint for Pose3DM_B** [here](https://drive.google.com/file/d/123AA9GDnnnbkiGuK-VoynY4bx4wPIn_1/view?usp=drive_link) and store it in the `./checkpoint` directory.

3. **Place your input videos** in the `./demo/video` directory.

4. **Run the demo script:**
   ```bash
   python vis.py --video AIG.mp4 --gpu 0
   ```

### Troubleshooting
- Ensure all required dependencies are correctly installed.
- Check that the correct CUDA version is being used.

## Copyright Notice

This project is licensed under the MIT License. For more details, please refer to the [LICENSE.txt](https://github.com/Reus3237/Pose3DM/blob/main/LICENSE.txt).

## Acknowledgements

Our code refers to the following repositories:
- [MotionBERT](https://github.com/Walter0807/MotionBERT)
- [MHFormer](https://github.com/Vegetebird/MHFormer)
- [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer)
- [PoseMamba](https://github.com/nankingjing/PoseMamba)

## Citation

If you find our work useful for your project, please consider citing our paper:

```bibtex
@article{Pose3DM2025,
  title={Bidirectional Mamba-enhanced 3D human pose estimation for accurate clinical gait analysis},
  author={Chengjun Wang, Wenhang Su, Jiabao Li, Jiahang Xu},
  journal={The Visual Computer},
  pages={1--31},
  year={2025},
  publisher={Springer}
}
```

## Links for Manuscript

To enhance transparency and reproducibility, please find links to our resources in the abstract of the paper.

- [Access the manuscript](<Insert_manuscript_link>)

