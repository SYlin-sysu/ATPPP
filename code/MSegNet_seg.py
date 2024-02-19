import torch
import argparse
import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from dense_unet import DenseUNet
from dataset import LiverDataset_whole
from tqdm import tqdm
import openslide
import shutil
import torch.nn.functional as F
from tools import gen_tissue_level_patch, gen_cell_level_patch
from losses import DICELossMultiClass,DICELoss,CE_Loss,Dice_Loss2
x_transforms = Compose([ToTensor(),Normalize([0.5, 0.5 ,0.5], [1, 1, 1])])
y_transforms = transforms.ToTensor()

def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
def walkFile(file):
    filepath_list = []
    for root, dirs, files in os.walk(file):
        for f in files:
            filepath_list.append(os.path.join(root, f))
    return filepath_list
val_interval = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_curve = list()
valid_curve = list()
def color_nor(img_dir,save_dir):
    from histomicstk.preprocessing.color_normalization import reinhard
    from histomicstk.saliency.tissue_detection import get_tissue_mask
    from skimage.transform import resize
    cnorm = {
        'mu': np.array([8.74108109, -0.12440419, 0.0444982]),
        'sigma': np.array([0.6135447, 0.10989545, 0.0286032]),
    }
    for imgname in os.listdir(img_dir):
        imgpath = os.path.join(img_dir, imgname)
        savepath = os.path.join(save_dir, imgname)
        img = cv2.imread(imgpath)
        try:
            mask_out, _ = get_tissue_mask(
                img, deconvolve_first=True,
                n_thresholding_steps=1, sigma=1.5, min_size=30)
        except ValueError:
            cv2.imwrite(savepath, img)
            continue
        mask_out = resize(
            mask_out == 0, output_shape=img.shape[:2],
            order=0, preserve_range=True) == 1

        img_nor = reinhard(img, target_mu=cnorm['mu'], target_sigma=cnorm['sigma'], mask_out= mask_out)
        cv2.imwrite(savepath, img_nor)
def get_slide_dic(slide_dir):
    slide_dic = {}
    image_dir = 'G:\linsy\LIHC\wsi_image'
    caseid2filepath = {}
    for dirname in os.listdir(slide_dir):
        dirpath = os.path.join(slide_dir, dirname)
        if os.path.isdir(dirpath):
            filepath_list = walkFile(dirpath)
            for filepath in filepath_list:
                if filepath.split('.')[-1] == 'svs':
                    submitter_id = "-".join(filepath.split('\\')[-1].split('-')[0:3])
                    if submitter_id not in caseid2filepath.keys():
                        caseid2filepath[submitter_id] = []
                        caseid2filepath[submitter_id].append(filepath)
                    else:
                        caseid2filepath[submitter_id].append(filepath)
    for submitter_id in os.listdir(image_dir):
        submitter_dir = os.path.join(image_dir, submitter_id)
        slide_list = [a.split('\\')[-1] for a in caseid2filepath[submitter_id]]
        slide_path = caseid2filepath[submitter_id][slide_list.index(os.listdir(submitter_dir)[0].replace('.png','.svs'))]
        slide_dic [submitter_id] = slide_path
    return slide_dic

def plot_pred(npy_data,save_path):
    img_array = np.zeros((npy_data.shape[0],npy_data.shape[1],3))
    label_list = ['kzxg', 'mgas', 'tumor']
    color = [[0,255,0],[0,0,255],[0,150,130],[0,0,0]]
    #color = [[0, 255, 0], [0, 150, 130], [0, 0, 0]]
    for x in range(npy_data.shape[0]):
        for y in range(npy_data.shape[1]):
            img_array[x][y] = color[npy_data[x][y]]

    cv2.imwrite(save_path,img_array)
def plot_pred_tissue(npy_data,save_path):
    img_array = np.zeros((256,256,3))
    color = [[0,255,0],[0,0,220],[0,150,130],[0,0,0]]
    #color = [[0, 255, 0], [0, 150, 130], [0, 0, 0]]
    for x in range(256):
        for y in range(256):
            img_array[x][y] = color[npy_data[x][y]]

    cv2.imwrite(save_path,img_array)
def plot_pred_L1(npy_data,save_path):
    img_array = np.zeros((npy_data.shape[0], npy_data.shape[1], 3))
    color = [[255, 255, 170], [127, 85, 85], [0, 85, 0], [0, 170, 255], [0, 0, 0]]
    color = [[0, 0, 0], [255, 255, 170], [127, 85, 85], [0, 85, 0]]
    # color = [[0, 255, 0], [0, 150, 130], [0, 0, 0]]
    for x in range(npy_data.shape[0]):
        for y in range(npy_data.shape[1]):
            img_array[x][y] = color[npy_data[x][y]]
    cv2.imwrite(save_path, img_array)

def plot_pred_merge(seg_result_L2_path,seg_result_L1_path, save_path):
    seg_result_L2 = cv2.imread(seg_result_L2_path)
    seg_result_L1 = cv2.imread(seg_result_L1_path)

    for x in range(seg_result_L1.shape[0]):
        for y in range(seg_result_L1.shape[1]):
            if np.sum(seg_result_L1[x][y])!=0:
                seg_result_L2[x][y] = seg_result_L1[x][y]
    cv2.imwrite(save_path, seg_result_L2)


def DenseUnet_seg(args, slide_path):
    mkdir(args.data_dir)
    mkdir(args.seg_results_dir)
    batch_size = args.batch_size
    model_L2 = DenseUNet(NUM_CLASSES_L2, downsample=True, pretrained_encoder_uri=None).to(device)
    model_L2.load_state_dict(torch.load(args.ckpt_L2, map_location='cuda'))
    model_L1 = DenseUNet(NUM_CLASSES_L1, downsample=True, pretrained_encoder_uri=None).to(device)
    model_L1.load_state_dict(torch.load(args.ckpt_L1, map_location='cuda'))
    n=0
    
    pid = os.path.basename(slide_path).split('.')[0]
    slide = openslide.open_slide(slide_path)
    x_slide, y_slide = slide.level_dimensions[2]
    resolution = int(slide.level_downsamples[2])
    tile_size = 256
    overlap_size = 128

    print("Segmenting tissue-level ROI...")
    gen_tissue_level_patch(tile_size, overlap_size, slide_path, args.data_dir, pid)
    pid_dir = os.path.join(args.data_dir, pid)
    patches_tissue_dir = os.path.join(pid_dir, 'patches_L2')
    seg_result_pid_dir = os.path.join(args.seg_results_dir, pid)
    mkdir(seg_result_pid_dir)
    seg_result_tissue_path = os.path.join(seg_result_pid_dir, pid+'_L2.png')
    foreground_tissue_path = os.path.join(pid_dir, 'foreground', pid+'_L2.npy')

    valid_dataset = LiverDataset_whole(patches_tissue_dir)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    x_index = int(x_slide / (tile_size - overlap_size))
    y_index = int(y_slide / (tile_size - overlap_size))
    pred_npy = np.ones((NUM_CLASSES_L2, y_slide, x_slide)) * -1000
    pred_npy[NUM_CLASSES_L2 - 1] += 1

    model_L2.eval()
    with torch.no_grad():
        step_val = 0
        for x, img_name in tqdm(val_loader):
            step_val += 1
            x = x.type(torch.FloatTensor)
            inputs = x.to(device)
            outputs = model_L2(inputs)
            # y = F.sigmoid(y)
            outputs = outputs.data.cpu().numpy()
            N, _, h, w = outputs.shape
            for i in range(N):
                filename = img_name[i]
                x = int(filename.split('.')[0].split('_')[0])
                y = int(filename.split('.')[0].split('_')[1])
                x_ = int(x / resolution)
                y_ = int(y / resolution)
                if x_ + tile_size <= x_slide and y_ + tile_size <= y_slide:
                    pred_npy[:, y_:y_ + tile_size, x_:x_ + tile_size] = np.maximum(outputs[i],
                                                                                    pred_npy[:, y_:y_ + tile_size,
                                                                                    x_:x_ + tile_size])
                elif x_ + tile_size > x_slide and y_ + tile_size > y_slide:
                    pred_npy[:, y_:y_slide, x_:x_slide] = np.maximum(
                        outputs[i, :, :y_slide - (y_ + tile_size), :x_slide - (x_ + tile_size)],
                        pred_npy[:, y_:y_slide, x_:x_slide])
                elif x_ + tile_size > x_slide:
                    pred_npy[:, y_:y_ + tile_size, x_:x_slide] = np.maximum(
                        outputs[i, :, :, :x_slide - (x_ + tile_size)], pred_npy[:, y_:y_ + tile_size, x_:x_slide])
                else:
                    pred_npy[:, y_:y_slide, x_:x_ + tile_size] = np.maximum(
                        outputs[i, :, :y_slide - (y_ + tile_size), :], pred_npy[:, y_:y_slide, x_:x_ + tile_size])
                # plot_pred_tissue(pred, save_patch_path)
        result_npy = pred_npy.transpose(1, 2, 0).reshape(-1, NUM_CLASSES_L2).argmax(axis=1).reshape(y_slide, x_slide)
        #np.save(pred_L2_path, result_npy)
        tissue_npy = np.load(foreground_tissue_path).T
        result_npy[tissue_npy == 0] = NUM_CLASSES_L2 - 1
        plot_pred(result_npy, seg_result_tissue_path)
    shutil.rmtree(patches_tissue_dir)

    print("Segmenting cell-level ROI...")
    gen_cell_level_patch(tile_size, overlap_size, slide_path, args.data_dir, args.seg_results_dir, pid)
    pid_dir = os.path.join(args.data_dir, pid)
    patches_cell_dir = os.path.join(pid_dir, 'patches_L1')
    seg_result_pid_dir = os.path.join(args.seg_results_dir, pid)
    seg_result_cell_path = os.path.join(seg_result_pid_dir, pid+'_L1.png')
    foreground_cell_path = os.path.join(pid_dir, 'foreground', pid+'_L1.npy')

    valid_dataset = LiverDataset_whole(patches_cell_dir)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    x_slide, y_slide = slide.level_dimensions[1]
    resolution = int(slide.level_downsamples[1])
    tile_size = 256
    overlap_size = 128
    # pred_npy = np.ones((NUM_CLASSES, y_slide, x_slide))*-float('inf')
    pred_npy = np.ones((NUM_CLASSES_L1, y_slide, x_slide), dtype=np.dtype('i2')) * (-1000)
    # pred_npy1 = np.zeros((y_slide, x_slide), dtype='int8')
    model_L1.eval()
    with torch.no_grad():
        step_val = 0
        for x, img_name in tqdm(val_loader):
            step_val += 1
            x = x.type(torch.FloatTensor)
            inputs = x.to(device)
            outputs = model_L1(inputs)
            # y = F.sigmoid(y)
            outputs = outputs.data.cpu().numpy()
            N, _, h, w = outputs.shape
            for i in range(N):
                filename = img_name[i]
                x = int(filename.split('.')[0].split('_')[0])
                y = int(filename.split('.')[0].split('_')[1])
                x_ = int(x / resolution)
                y_ = int(y / resolution)

                save_path = os.path.join('G:\linsy\LIHC\input_patch_seg\seg_result',filename)
                #pred = outputs[i].transpose(1, 2, 0).reshape(-1, NUM_CLASSES_L1).argmax(axis=1).reshape(h, w)
                #plot_pred_L1(pred, save_path)
                if x_ + tile_size <= x_slide and y_ + tile_size <= y_slide:
                    pred_npy[:, y_:y_ + tile_size, x_:x_ + tile_size] = np.maximum(outputs[i],
                                                                                    pred_npy[:, y_:y_ + tile_size,
                                                                                    x_:x_ + tile_size])
                elif x_ + tile_size > x_slide and y_ + tile_size > y_slide:
                    pred_npy[:, y_:y_slide, x_:x_slide] = np.maximum(
                        outputs[i, :, :y_slide - (y_ + tile_size), :x_slide - (x_ + tile_size)],
                        pred_npy[:, y_:y_slide, x_:x_slide])
                elif x_ + tile_size > x_slide:
                    pred_npy[:, y_:y_ + tile_size, x_:x_slide] = np.maximum(
                        outputs[i, :, :, :x_slide - (x_ + tile_size)], pred_npy[:, y_:y_ + tile_size, x_:x_slide])
                else:
                    pred_npy[:, y_:y_slide, x_:x_ + tile_size] = np.maximum(
                        outputs[i, :, :y_slide - (y_ + tile_size), :], pred_npy[:, y_:y_slide, x_:x_ + tile_size])
        pred_npy = pred_npy.astype('i1')
        pred_npy = pred_npy.transpose(1, 2, 0).astype('i1').reshape(-1, NUM_CLASSES_L1).astype('i1').argmax(axis=1).reshape(
            pred_npy.shape[1], pred_npy.shape[2])
        #np.save(pred_L1_path, pred_npy)

        pred_npy = pred_npy.astype(np.uint8)
        resolution = 0.25
        pred_npy = cv2.resize(pred_npy, (0, 0), fx=resolution, fy=resolution, interpolation=cv2.INTER_AREA)
        plot_pred_L1(pred_npy, seg_result_cell_path)
    shutil.rmtree(patches_cell_dir)

    seg_result_merge_path = os.path.join(seg_result_pid_dir, pid+'_seg_result.png')
    plot_pred_merge(seg_result_tissue_path, seg_result_cell_path, seg_result_merge_path)


if __name__ == '__main__':
    NUM_CLASSES_L2 = 4
    NUM_CLASSES_L1 = 4

    parse = argparse.ArgumentParser()
    parse.add_argument("--batch_size", type=int, default=32)
    #parse.add_argument("--device_ids", type=str, default='0')
    parse.add_argument("--ckpt_L2", type=str, help="the path of the folder of tissue-level segmentation model", default="./model/L2_model.pth")
    parse.add_argument("--ckpt_L1", type=str, help="the path of the folder of cell-level segmentation model", default="./model/L1_model.pth")
    parse.add_argument("--slide_dir", type=str, help="the path of the folder of input WSIs", default="./data/slide")
    parse.add_argument("--data_dir", type=str, help="the path of the folder of intermediate data", default="./data/intermediate_data")
    parse.add_argument("--seg_results_dir", type=str, help="the path of the folder of segmentation result", default="./data/seg_result")
    args = parse.parse_args()

    for filename in os.listdir(args.slide_dir):
        slide_path = os.path.join(args.slide_dir, filename)
        pid = os.path.basename(slide_path).split('.')[0]
        print(pid)
        if pid=='TCGA-2Y-A9GY-01A':
            continue
        pid_dir = os.path.join(args.data_dir, pid)
        DenseUnet_seg(args, slide_path)

