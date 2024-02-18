from torch.utils.data import Dataset
import PIL.Image as Image
import os,shutil
import cv2
import torch
import numpy as np
import random
import openslide
from tqdm import tqdm
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from scipy.interpolate import interp1d
from openslide.deepzoom import DeepZoomGenerator
from collections import Counter

def TissueMaskGenerationOS(slide_obj, level, RGB_min=50):
    img_RGB = slide_obj.read_region((0, 0),level,slide_obj.level_dimensions[level])
    img_RGB = np.transpose(np.array(img_RGB.convert('RGB')),axes=[1,0,2])
    img_HSV = rgb2hsv(img_RGB)
    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    # r = img_RGB[:,:,0] < 235
    # g = img_RGB[:,:,1] < 210
    # b = img_RGB[:,:,2] < 235
    # tissue_mask = np.logical_or(r,np.logical_or(g,b))
    return tissue_mask
def BinMorphoProcessMaskOS(mask, level):
    """
    Binary operation performed on tissue mask
    """
    close_kernel = np.ones((20, 20), dtype=np.uint8)
    image_close = cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((5, 5), dtype=np.uint8)
    image_open = cv2.morphologyEx(np.array(image_close), cv2.MORPH_OPEN, open_kernel)
    #TTD - Come up with better dilation strategy
    if level ==1:
        kernel = np.ones((70, 70), dtype=np.uint8)
    elif level == 2:
        kernel = np.ones((60, 60), dtype=np.uint8)
    elif level == 3:
        kernel = np.ones((35, 35), dtype=np.uint8)
    elif level == 4:
        kernel = np.ones((10, 10), dtype=np.uint8)
    else:
        print(level)
        raise ValueError("Kernel for this level not fixed")
    image = cv2.dilate(image_open,kernel,iterations = 1)
    kernel = np.ones((30, 30), dtype=np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    return image

def FillHole(im_in, SaveNpyPath, png_save_path):
    #im_in = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("im_in.png", im_in)
    im_floodfill = im_in.copy()

    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    isbreak = False
    for i in range(im_floodfill.shape[0]):
        for j in range(im_floodfill.shape[1]):
            if (im_floodfill[i][j] == 0):
                seedPoint = (i, j)
                isbreak = True
                break
        if (isbreak):
            break

    cv2.floodFill(im_floodfill, mask, seedPoint, 255)

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    im_out = im_in | im_floodfill_inv
    np.save(SaveNpyPath, im_out)
    cv2.imwrite(png_save_path, im_out.T)

def walkFile(file):
    filepath_list = []
    for root, dirs, files in os.walk(file):
        for f in files:
            filepath_list.append(os.path.join(root, f))
    return filepath_list

def get_slide_dic(slide_dir):
    slide_dic = {}
    image_dir = 'H:\LIHC\wsi_image'
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

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def gen_tissue_level_patch(tile_size, overlap_size, slide_path, data_dir, pid):
    slide = openslide.open_slide(slide_path)
    pid_dir = os.path.join(data_dir, pid)
    mkdir(pid_dir)

    foreground_dir = os.path.join(pid_dir, 'foreground')
    mkdir(foreground_dir)
    png_save_path = os.path.join(foreground_dir, pid + '_L2.png')
    npy_save_path = os.path.join(foreground_dir, pid + '_L2.npy')
    target_mask = TissueMaskGenerationOS(slide, 2)
    target_mask = BinMorphoProcessMaskOS(np.uint8(target_mask), 4)
    target_mask = FillHole(target_mask, npy_save_path, png_save_path)
    target_mask = cv2.imread(png_save_path, cv2.IMREAD_GRAYSCALE).T

    patch_dir = os.path.join(pid_dir, 'patches_L2')
    mkdir(patch_dir)
    strided_mask = np.ones_like(target_mask)
    ones_mask = np.zeros_like(target_mask)
    factor = tile_size - overlap_size
    ones_mask[::factor, ::factor] = strided_mask[::factor, ::factor]
    strided_mask = ones_mask * target_mask
    X_idcs, Y_idcs = np.where(strided_mask)
    idcs_num = len(X_idcs)
    for idx in range(idcs_num):
        x_coord, y_coord = X_idcs[idx], Y_idcs[idx]
        if y_coord - factor>=0:
            strided_mask[x_coord, y_coord - factor] = 1
        if y_coord + factor<= strided_mask.shape[1]:
            strided_mask[x_coord, y_coord + factor] = 1
        if x_coord - factor>=0:
            strided_mask[x_coord - factor, y_coord] = 1
        if x_coord + factor<= strided_mask.shape[0]:
            strided_mask[x_coord + factor, y_coord] = 1
    X_idcs, Y_idcs = np.where(strided_mask)
    idcs_num = len(X_idcs)
    print('Idcs_num: %d' % (idcs_num), pid)

    x_slide, y_slide = slide.level_dimensions[2]
    resolution = int(slide.level_downsamples[2])
    for idx in tqdm(range(idcs_num)):
        x_coord, y_coord = X_idcs[idx], Y_idcs[idx]
        x_max_dim, y_max_dim = slide.level_dimensions[0]

        # x = int(x_coord * self._resolution)
        # y = int(y_coord * self._resolution)
        x = int(x_coord * resolution - tile_size // 2)
        y = int(y_coord * resolution - tile_size // 2)

        # If Image goes out of bounds
        x = max(0, min(x, x_max_dim - tile_size))
        y = max(0, min(y, y_max_dim - tile_size))

        # Converting pil image to np array transposes the w and h
        img = slide.read_region((x, y), 2, (tile_size, tile_size)).convert('RGB')
        save_path = os.path.join(patch_dir, str(x) + '_' + str(y) + '.png')
        img.save(save_path)

def gen_cell_level_patch(tile_size, overlap_size, slide_path, data_dir, result_dir, pid):
    slide = openslide.open_slide(slide_path)
    pid_dir = os.path.join(data_dir, pid)
    mkdir(pid_dir)

    foreground_dir = os.path.join(pid_dir, 'foreground')
    mkdir(foreground_dir)
    foreground_L2_png_path = os.path.join(foreground_dir, pid + '_L2.png')
    foreground_L1_npy_path = os.path.join(foreground_dir, pid + '_L2.npy')
    foreground_L1_png_path = os.path.join(foreground_dir, pid + '_L1.png')
    foreground_L2_img = cv2.imread(foreground_L2_png_path)
    foreground_L1_img = cv2.resize(foreground_L2_img, (0, 0), fx=int(slide.level_downsamples[2] / slide.level_downsamples[1]),
                               fy=int(slide.level_downsamples[2] / slide.level_downsamples[1]),
                               interpolation=cv2.INTER_AREA)
    cv2.imwrite(foreground_L1_png_path, foreground_L1_img)

    seg_result_pid_dir = os.path.join(result_dir, pid)
    seg_result_tissue_L2_path = os.path.join(seg_result_pid_dir, pid+'_L2.png')
    seg_result_tissue_L1_path = os.path.join(seg_result_pid_dir, pid+'_mask_L1.png')
    seg_result_tissue_L2_img = cv2.imread(seg_result_tissue_L2_path)
    seg_result_tissue_L1_img = cv2.resize(seg_result_tissue_L2_img, (0, 0), fx=int(slide.level_downsamples[2] / slide.level_downsamples[1]),
                               fy=int(slide.level_downsamples[2] / slide.level_downsamples[1]),
                               interpolation=cv2.INTER_AREA)
    cv2.imwrite(seg_result_tissue_L1_path, seg_result_tissue_L1_img)

    L1_mask = cv2.imread(seg_result_tissue_L1_path,cv2.IMREAD_GRAYSCALE)
    tissue_mask = cv2.imread(foreground_L1_png_path,cv2.IMREAD_GRAYSCALE)
    inv_L1_mask = np.where(L1_mask,0,255)
    target_mask = (tissue_mask&inv_L1_mask).T

    strided_mask = np.ones_like(target_mask,dtype=bool)
    ones_mask = np.zeros_like(target_mask,dtype=bool)
    factor = tile_size - overlap_size
    ones_mask[::factor, ::factor] = strided_mask[::factor, ::factor]
    strided_mask = ones_mask * target_mask
    X_idcs, Y_idcs = np.where(strided_mask)
    idcs_num = len(X_idcs)
    print('Idcs_num %d' % (idcs_num))

    patch_dir = os.path.join(pid_dir, 'patches_L1')
    mkdir(patch_dir)

    x_slide, y_slide = slide.level_dimensions[1]
    resolution = int(slide.level_downsamples[1])
    x_index = int(x_slide / (tile_size - overlap_size))
    y_index = int(y_slide / (tile_size - overlap_size))
    for idx in tqdm(range(idcs_num)):
        x_coord, y_coord = X_idcs[idx], Y_idcs[idx]
        x_max_dim, y_max_dim = slide.level_dimensions[0]

        # x = int(x_coord * self._resolution)
        # y = int(y_coord * self._resolution)
        x = int(x_coord * resolution - tile_size // 2)
        y = int(y_coord * resolution - tile_size // 2)

        # If Image goes out of bounds
        x = max(0, min(x, x_max_dim - tile_size))
        y = max(0, min(y, y_max_dim - tile_size))

        # Converting pil image to np array transposes the w and h
        img = slide.read_region((x, y), 1, (tile_size, tile_size)).convert('RGB')
        save_path = os.path.join(patch_dir, str(x) + '_' + str(y) + '.png')
        img.save(save_path)


def get_predAtissue_L1_png(result_dir, tissue_dir,slide, pid):
    L2_img_path = os.path.join(result_dir, pid + '_L2.png')
    L2_img = cv2.imread(L2_img_path)
    L1_img = cv2.resize(L2_img, (0, 0), fx=int(slide.level_downsamples[2] / slide.level_downsamples[1]),
                        fy=int(slide.level_downsamples[2] / slide.level_downsamples[1]),
                        interpolation=cv2.INTER_AREA)
    L1_img_path = os.path.join(result_dir, pid + '_L1.png')
    cv2.imwrite(L1_img_path, L1_img)

    tissue_L2_img_path = os.path.join(tissue_dir, pid + '_L2.png')
    tissue_L2_img = cv2.imread(tissue_L2_img_path)
    tissue_L1_img = cv2.resize(tissue_L2_img, (0, 0), fx=int(slide.level_downsamples[2] / slide.level_downsamples[1]),
                               fy=int(slide.level_downsamples[2] / slide.level_downsamples[1]),
                               interpolation=cv2.INTER_AREA)
    tissue_L1_img_path = os.path.join(tissue_dir, pid + '_L1.png')
    cv2.imwrite(tissue_L1_img_path, tissue_L1_img)

def gen_overlap_L1_patch(slide_dir,patch_root,tissue_dir,result_dir,pid):
    tile_size = 256
    overlap_size = 64
    slide_dct = get_slide_dic(slide_dir)
    slide_path = slide_dct[pid]
    slide = openslide.open_slide(slide_path)
    patch_dir = os.path.join(patch_root,pid)
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)
    get_predAtissue_L1_png(result_dir, tissue_dir, slide, pid)
    L1_img_path =  os.path.join(result_dir, pid + '_L1.png')
    tissue_L1_img_path = os.path.join(tissue_dir, pid + '_L1.png')

    L1_mask = cv2.imread(L1_img_path,cv2.IMREAD_GRAYSCALE)
    tissue_mask = cv2.imread(tissue_L1_img_path,cv2.IMREAD_GRAYSCALE)
    inv_L1_mask = np.where(L1_mask,0,255)
    target_mask = (tissue_mask&inv_L1_mask).T
    #cv2.imwrite(os.path.join('E:\Study\lab\\file\input_patch_seg6\save', pid+'_gray.png'),target_mask)

    strided_mask = np.ones_like(target_mask,dtype=bool)
    ones_mask = np.zeros_like(target_mask,dtype=bool)
    factor = tile_size - overlap_size
    ones_mask[::factor, ::factor] = strided_mask[::factor, ::factor]
    strided_mask = ones_mask * target_mask
    X_idcs, Y_idcs = np.where(strided_mask)
    idcs_num = len(X_idcs)
    print('Idcs_num %d' % (idcs_num))

    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)
    x_slide, y_slide = slide.level_dimensions[1]
    resolution = int(slide.level_downsamples[1])
    x_index = int(x_slide / (tile_size - overlap_size))
    y_index = int(y_slide / (tile_size - overlap_size))
    for idx in range(idcs_num):
        x_coord, y_coord = X_idcs[idx], Y_idcs[idx]
        x_max_dim, y_max_dim = slide.level_dimensions[0]

        # x = int(x_coord * self._resolution)
        # y = int(y_coord * self._resolution)
        x = int(x_coord * resolution - tile_size // 2)
        y = int(y_coord * resolution - tile_size // 2)

        # If Image goes out of bounds
        x = max(0, min(x, x_max_dim - tile_size))
        y = max(0, min(y, y_max_dim - tile_size))

        # Converting pil image to np array transposes the w and h
        img = slide.read_region((x, y), 1, (tile_size, tile_size)).convert('RGB')
        save_path = os.path.join(patch_dir, str(x) + '_' + str(y) + '.png')
        img.save(save_path)