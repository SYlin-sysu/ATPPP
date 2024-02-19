# ATPPP (tumour & proximal paracancerous-based pipeline for prognosis prediction)
Applying image features of proximal paracancerous tissues in predicting prognosis of patients with hepatocellular carcinoma
## Pre-requisites:
+ NVIDIA GPU (Nvidia GeForce RTX 2080 Ti) with CUDA 11.6
+ pytorch(1.8.1), torchvision(0.9.1), opencv-python(4.5.4.58), scikit-image(0.17.2), numpy(1.19.2), pandas(1.1.5), matplotlib(3.3.4), openslide-python(1.1.2), mahotas(1.4.12), scipy(1.5.4), pillow(8.4.0), tqdm(4.62.3)
## Usage 
### 1. Segmenting WSI using MSegNet
```
python ./code/MSegNet_seg.py --batch_size 32 --ckpt_L1 ./model/L1_model.pth --ckpt_L2 ./model/L2_model.pth --slide_dir ./data/slide --data_dir ./data/intermediate_data --seg_results_dir ./data/seg_result
```
### 2. Calculating tumour & proximal paracancerous-based features
```
python ./code/get_features.py --slide_dir ./data/slide --data_dir ./data/intermediate_data --seg_results_dir ./data/seg_result --save_csv_path ./data/image_results.csv
```
