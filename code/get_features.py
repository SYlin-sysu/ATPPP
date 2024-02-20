import sys
import os
import random
from scipy import stats
import argparse
import logging
import time
from shutil import copyfile
from multiprocessing import Pool, Value, Lock
import matplotlib.pyplot as plt
import openslide
import cv2
import mahotas
import numpy as np
import pandas as pd
from tqdm import tqdm

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_pid_patch2(pid, wsi_path, mask, save_dir,image_size=256,slide_level=0, sampling_stride=16, roi_masking=True,pos=0,sample=250):
    slide = openslide.OpenSlide(wsi_path)
    #print(slide.level_dimensions, sampling_stride, slide.level_downsamples)
    level = len(slide.level_dimensions) - 1
    X_slide, Y_slide = slide.level_dimensions[slide_level]
    sampling_stride = sampling_stride // (X_slide // int(slide.level_dimensions[level][0]))
    X_slide_level_dim, Y_slide_level_dim = slide.level_dimensions[level]
    '''
    print("Level %d (Highest level %d) , stride %d" % (
        level, slide.level_count - 1, sampling_stride))
    print("Actual dimensions: (%d,%d)" % (X_slide, Y_slide))
    print("Level dimensions: (%d,%d)" % (X_slide_level_dim, Y_slide_level_dim))
    '''
    factor = sampling_stride
    # self._level = len(self._slide.level_dimensions) - 1
    # self._sampling_stride = self._sampling_stride / (2.0**self._level)
    if pos ==1:
        mask[:int(mask.shape[0]/2), :] = 0
    elif pos ==0:
        mask[int(mask.shape[0] / 2): , :] = 0
    # self._all_bbox_mask = get_all_bbox_masks(self._mask, factor)
    # self._largest_bbox_mask = find_largest_bbox(self._mask, factor)
    # self._all_strided_bbox_mask = get_all_bbox_masks_with_stride(self._mask, factor)
    X_mask, Y_mask = mask.shape
    # print (self._mask.shape, np.where(self._mask>0))
    # imshow(self._mask.T)
    # cm17 dataset had issues with images being power's of 2 precisely
    # if X_slide != X_mask or Y_slide != Y_mask:
    #print('Mask (%d,%d) and Slide(%d,%d) ' % (X_mask, Y_mask, X_slide, Y_slide))
    if X_slide // X_mask != Y_slide // Y_mask:
        raise Exception('Slide/Mask dimension does not match ,'
                        ' X_slide / X_mask : {} / {},'
                        ' Y_slide / Y_mask : {} / {}'
                        .format(X_slide, X_mask, Y_slide, Y_mask))

    resolution = np.round(X_slide * 1.0 / X_mask) * int(slide.level_downsamples[slide_level])
    #print('Resolution (%d)' % (resolution))
    if not np.log2(resolution).is_integer():
        raise Exception('Resolution (X_slide / X_mask) is not power of 2 :'
                        ' {}'.format(resolution))

    # all the idces for tissue region from the tissue mask
    strided_mask = np.ones_like(mask)
    ones_mask = np.zeros_like(mask)
    ones_mask[::factor, ::factor] = strided_mask[::factor, ::factor]

    if roi_masking:
        strided_mask = ones_mask * mask
        # self._strided_mask = ones_mask*self._largest_bbox_mask
        # self._strided_mask = ones_mask*self._all_bbox_mask
        # self._strided_mask = self._all_strided_bbox_mask
    else:
        strided_mask = ones_mask
        # print (np.count_nonzero(self._strided_mask), np.count_nonzero(self._mask[::factor, ::factor]))
    # imshow(self._strided_mask.T, self._mask[::factor, ::factor].T)
    # imshow(self._mask.T, self._strided_mask.T)
    X_idcs, Y_idcs = np.where(strided_mask)
    idcs_num = len(X_idcs)
    print('Idcs_num: %d' % (idcs_num))
    if sample and idcs_num >sample:
        idcs = list(zip(X_idcs, Y_idcs))
        idcs = np.array(random.sample(idcs, sample))
        X_idcs = idcs[:,0]
        Y_idcs = idcs[:, 1]
        idcs_num = sample
    for idx in tqdm(range(idcs_num)):
        x_coord, y_coord = X_idcs[idx], Y_idcs[idx]
        x_max_dim, y_max_dim = slide.level_dimensions[0]

        # x = int(x_coord * self._resolution)
        # y = int(y_coord * self._resolution)
        x = int(x_coord * resolution - image_size // 2)
        y = int(y_coord * resolution - image_size // 2)

        # If Image goes out of bounds
        x = max(0, min(x, x_max_dim - image_size))
        y = max(0, min(y, y_max_dim - image_size))

        # Converting pil image to np array transposes the w and h
        img = slide.read_region((x, y), slide_level, (image_size, image_size)).convert('RGB')
        img.save(os.path.join(save_dir, pid + '_' + str(x) + "_" + str(y) + '.png'))

def separate_png(filepath,classname):
    arr = cv2.imread(filepath)
    save_dir = '/data2/users/linsy/HE_liver_cancer/CAMELYON-master1/output/temp'
    classname_list = ['DVLE', 'MVI', 'TUM','DR', 'ICI', 'HS']
    color = [[0, 255, 0], [0, 0, 255], [0, 150, 130], [255, 255, 170], [127, 85, 85], [0, 85, 0]]
    color_sum = []
    for a in color:
        color_sum.append(a[0]+a[1]+2*a[2])
    classname2rgb = dict(zip(classname_list,color_sum))
    r_img, g_img, b_img = arr[:, :, 0].copy(), arr[:, :, 1].copy(), arr[:, :, 2].copy()
    img = r_img +  g_img + 2*b_img
    dst = np.where(img == classname2rgb[classname]%256, 1, 0)
    #cv2.imwrite(os.path.join(save_dir,filepath.split('/')[-1].split('.')[0]+'_'+classname+'.png'), dst*255)
    return (dst*255).astype('uint8').T

def get_patch_500(slide_path, pid_dir, seg_result_dir):
    pid = os.path.basename(slide_path).split('.')[0]
    save_patch_dir = os.path.join(pid_dir, 'patch_500')
    makedir(save_patch_dir)
    class_name_list = ['DR', 'ICI']
    for class_name in class_name_list:
        slide = openslide.OpenSlide(slide_path)
        seg_result_tissue_path = os.path.join(seg_result_dir, pid + '_L2.png')
        seg_result_cell_path = os.path.join(seg_result_dir, pid + '_L1.png')
        print(class_name, pid)
        makedir(os.path.join(save_patch_dir, class_name))

        if class_name in ['DVLE', 'MVI', 'TUM']:
            mask = separate_png(seg_result_tissue_path, class_name)
        else:
            mask = separate_png(seg_result_cell_path, class_name)
        level = 1
        X_slide, Y_slide = slide.level_dimensions[2]
        X_mask, Y_mask = mask.shape
        if X_mask != X_slide:
            if X_slide > X_mask:
                for _ in range(X_slide - X_mask):
                    mask = np.insert(mask, 0, values=1, axis=0)
            else:
                for _ in range(X_mask - X_slide):
                    mask = np.delete(mask, 0, axis=0)
        if Y_mask != Y_slide:
            if Y_slide > Y_mask:
                for _ in range(Y_slide - Y_mask):
                    mask = np.insert(mask, 0, values=1, axis=1)
            else:
                for _ in range(Y_mask - Y_slide):
                    mask = np.delete(mask, 0, axis=1)
        mask = cv2.resize(mask, (0, 0), fx=int(slide.level_downsamples[2] / slide.level_downsamples[1]),
                            fy=int(slide.level_downsamples[2] / slide.level_downsamples[1]),
                            interpolation=cv2.INTER_AREA)
        get_pid_patch2(pid, slide_path, mask.copy(),
                        os.path.join(save_patch_dir, class_name), image_size=500, slide_level=level,
                        sampling_stride=1500,
                        roi_masking=True, pos=2,sample=None)
        

def get_patch_2000(slide_path, pid_dir, seg_result_dir):
    pid = os.path.basename(slide_path).split('.')[0]
    save_patch_dir = os.path.join(pid_dir, 'patch_2000')
    makedir(save_patch_dir)
    class_name_list = ['TUM']
    for class_name in class_name_list:
        slide = openslide.OpenSlide(slide_path)
        seg_result_tissue_path = os.path.join(seg_result_dir, pid + '_L2.png')
        seg_result_cell_path = os.path.join(seg_result_dir, pid + '_L1.png')

        print(class_name, pid)
        makedir(os.path.join(save_patch_dir, class_name))

        if class_name in ['DVLE', 'MVI', 'TUM']:
            mask = separate_png(seg_result_tissue_path, class_name)
        else:
            mask = separate_png(seg_result_cell_path, class_name)
        level = 1
        X_slide, Y_slide = slide.level_dimensions[2]
        X_mask, Y_mask = mask.shape
        if X_mask != X_slide:
            if X_slide > X_mask:
                for _ in range(X_slide - X_mask):
                    mask = np.insert(mask, 0, values=1, axis=0)
            else:
                for _ in range(X_mask - X_slide):
                    mask = np.delete(mask, 0, axis=0)
        if Y_mask != Y_slide:
            if Y_slide > Y_mask:
                for _ in range(Y_slide - Y_mask):
                    mask = np.insert(mask, 0, values=1, axis=1)
            else:
                for _ in range(Y_mask - Y_slide):
                    mask = np.delete(mask, 0, axis=1)
        mask = cv2.resize(mask, (0, 0), fx=int(slide.level_downsamples[2] / slide.level_downsamples[1]),
                            fy=int(slide.level_downsamples[2] / slide.level_downsamples[1]),
                            interpolation=cv2.INTER_AREA)
        get_pid_patch2(pid, slide_path, mask.copy(),
                        os.path.join(save_patch_dir, class_name), image_size=2000, slide_level=level,
                        sampling_stride=6000,
                        roi_masking=True, pos=2,sample=None)

def Nucleus_seg(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #print(cv2.mean(gray))
    ret, thresh = cv2.threshold(gray,cv2.mean(gray[gray<220])[0]-15, 255, cv2.THRESH_BINARY_INV )
    #thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 31, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel2, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)  # sure background area
    sure_fg = cv2.erode(thresh, kernel2, iterations=2)  # sure foreground area
    unknown = cv2.subtract(sure_bg, sure_fg)  # unknown area
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)

    img[markers == -1] = [0, 255, 0]
    markers_copy = markers.copy()
    markers_copy[markers_copy==-1] = 255
    markers_copy[markers_copy==1] = 0
    markers_copy[markers_copy >1] = 255
    markers_copy[0:-1,0] = 0
    markers_copy[0:-1,-1] = 0
    markers_copy[0,0:-1] = 0
    markers_copy[ -1,0:-1] = 0
    markers_copy = np.uint8(markers_copy)
    contours, hierarchy = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours),img

def cell_seg_patch_500(slide_path, pid_dir):
    pid = os.path.basename(slide_path).split('.')[0]
    save_patch_dir = os.path.join(pid_dir, 'patch_500')
    save_CDM_dir = os.path.join(pid_dir, 'patch_500_seg_array')
    #save_seg_result_dir = os.path.join(pid_dir, 'patch_500_cell_seg_result')

    makedir(save_CDM_dir)
    #makedir(save_seg_result_dir)
    tile_width = 10
    tile_size = 50
    class_name = 'ICI'
    save_class_patch_dir = os.path.join(save_patch_dir, class_name)
    if os.path.exists(save_class_patch_dir):
        print(pid, class_name, 'Idcs_num: %d' % (len(os.listdir(save_class_patch_dir))))
        class_CDM_dir = os.path.join(save_CDM_dir, class_name)
        #class_seg_result_dir = os.path.join(save_seg_result_dir, class_name)

        makedir(class_CDM_dir)
        #makedir(class_seg_result_dir)
        for patch in tqdm(os.listdir(save_class_patch_dir)):
            patch_path = os.path.join(save_class_patch_dir,patch)
            img = cv2.imread(patch_path)
            tile_array = np.zeros((tile_width, tile_width))
            array_name = patch.split('.')[0]+'.npy'
            array_path = os.path.join(class_CDM_dir, array_name)
            for a in range(tile_width):
                for b in range(tile_width):
                    subimg = img[a*tile_size:(a+1)*tile_size, b*tile_size:(b+1)*tile_size]
                    cell_num,seg_result = Nucleus_seg(subimg)
                    tile_array[a][b] = cell_num
                    #save_path = os.path.join(class_seg_result_dir,str(a)+'_'+str(b)+'.jpg')
                    #cv2.imwrite(save_path, seg_result)
            np.save(array_path, tile_array)

def cell_seg_patch_2000(slide_path, pid_dir):
    pid = os.path.basename(slide_path).split('.')[0]
    save_patch_dir = os.path.join(pid_dir, 'patch_2000')
    save_CDM_dir = os.path.join(pid_dir, 'patch_2000_seg_array')
    #save_seg_result_dir = os.path.join(pid_dir, 'patch_2000_cell_seg_result')

    makedir(save_CDM_dir)
    #makedir(save_seg_result_dir)
    tile_width = 20
    tile_size = 100
    class_name = 'TUM'
    save_class_patch_dir = os.path.join(save_patch_dir, class_name)
    if os.path.exists(save_class_patch_dir):
        print(pid, class_name, 'Idcs_num: %d' %len(os.listdir(save_class_patch_dir)))
        class_CDM_dir = os.path.join(save_CDM_dir, class_name)
        #class_seg_result_dir = os.path.join(save_seg_result_dir, class_name)

        makedir(class_CDM_dir)
        #makedir(class_seg_result_dir)
        for patch in tqdm(os.listdir(save_class_patch_dir)):
            patch_path = os.path.join(save_class_patch_dir,patch)
            img = cv2.imread(patch_path)
            tile_array = np.zeros((tile_width, tile_width))
            array_name = patch.split('.')[0]+'.npy'
            array_path = os.path.join(class_CDM_dir, array_name)
            for a in range(tile_width):
                for b in range(tile_width):
                    subimg = img[a*tile_size:(a+1)*tile_size, b*tile_size:(b+1)*tile_size]
                    cell_num,seg_result = Nucleus_seg(subimg)
                    tile_array[a][b] = cell_num
                    #save_path = os.path.join(class_seg_result_dir,str(a)+'_'+str(b)+'.jpg')
                    #cv2.imwrite(save_path, seg_result)
            np.save(array_path, tile_array)

def get_cell_density_haralick(npy_dir):
    Glcm1_array = []
    Glcm2_array = []
    Glcm3_array = []
    Glcm4_array = []
    for npy_name in os.listdir(npy_dir):
        npy_path = os.path.join(npy_dir,npy_name)
        npy_data = np.load(npy_path)
        try:
            haralick1 = mahotas.features.haralick(npy_data.astype('int'), True)[:2].mean(0)
        except ValueError:
            print("ValueError")
            continue
        try:
            haralick2 = mahotas.features.haralick(npy_data.astype('int'), True)[2:].mean(0)
        except ValueError:
            print("ValueError")
            continue
        Glcm1_array.append(haralick1)
        Glcm2_array.append(haralick2)
    Glcm1_array = np.array(Glcm1_array)
    Glcm2_array = np.array(Glcm2_array)
    result4= []
    for Glcm_array in [Glcm1_array, Glcm2_array]:
        result = []
        skew = []
        kurt = []
        for a in range(Glcm_array.shape[1]):
            s = pd.Series(Glcm_array[:,a])
            skew.append(stats.skew(Glcm_array[:,a]))
            kurt.append(stats.kurtosis(Glcm_array[:,a]))
        result.append(list(np.mean(Glcm_array,0)))
        result.append(list(np.median(Glcm_array,0)))
        result.append(list(np.var(Glcm_array,0)))
        result.append(kurt)
        result.append(skew)
        result4.append(result)
    return result4


def get_cell_density_haralick2(npy_dir):
    Glcm1_array = []
    Glcm2_array = []
    Glcm3_array = []
    Glcm4_array = []
    for npy_name in os.listdir(npy_dir):
        npy_path = os.path.join(npy_dir,npy_name)
        npy_data = np.load(npy_path)
        try:
            haralick = mahotas.features.haralick(npy_data.astype('int'), True)
        except ValueError:
            print("ValueError "+npy_name)
            continue
        Glcm1_array.append(haralick[0])
        Glcm2_array.append(haralick[1])
        Glcm3_array.append(haralick[2])
        Glcm4_array.append(haralick[3])
    Glcm1_array = np.array(Glcm1_array)
    Glcm2_array = np.array(Glcm2_array)
    Glcm3_array = np.array(Glcm3_array)
    Glcm4_array = np.array(Glcm4_array)
    result4= []
    for Glcm_array in [Glcm1_array, Glcm2_array,Glcm3_array,Glcm4_array]:
        result = []
        skew = []
        kurt = []
        for a in range(Glcm_array.shape[1]):
            s = pd.Series(Glcm_array[:,a])
            skew.append(stats.skew(Glcm_array[:,a]))
            kurt.append(stats.kurtosis(Glcm_array[:,a]))
        result.append(list(np.mean(Glcm_array,0)))
        result.append(list(np.median(Glcm_array,0)))
        result.append(list(np.var(Glcm_array,0)))
        result.append(kurt)
        result.append(skew)
        result4.append(result)
    return result4

def get_texture_haralick(image_dir):
    Glcm1_array = []
    Glcm2_array = []
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir,image_name)
        npy_data = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        try:
            haralick1 = mahotas.features.haralick(npy_data.astype('int'), True)[:2].mean(0)
        except ValueError:
            print("ValueError")
            continue
        try:
            haralick2 = mahotas.features.haralick(npy_data.astype('int'), True)[2:].mean(0)
        except ValueError:
            print("ValueError")
            continue
        Glcm1_array.append(haralick1)
        Glcm2_array.append(haralick2)
    Glcm1_array = np.array(Glcm1_array)
    Glcm2_array = np.array(Glcm2_array)
    result4= []
    for Glcm_array in [Glcm1_array, Glcm2_array]:
        result = []
        skew = []
        kurt = []
        for a in range(Glcm_array.shape[1]):
            s = pd.Series(Glcm_array[:,a])
            skew.append(s.skew())
            kurt.append(s.kurt())
        result.append(list(np.mean(Glcm_array,0)))
        result.append(list(np.median(Glcm_array,0)))
        result.append(list(np.var(Glcm_array,0)))
        result.append(kurt)
        result.append(skew)
        result4.append(result)
    return result4

def get_cell_density_features_ICI(slide_path, pid_dir):
    class_name_list =['ICI']
    pid = os.path.basename(slide_path).split('.')[0]
    CDM_dir = os.path.join(pid_dir, 'patch_500_seg_array')
    save_features_dir = os.path.join(pid_dir, 'features')

    makedir(save_features_dir)
    for class_name in class_name_list:
        class_CDM_dir = os.path.join(CDM_dir, class_name)
        #print(pid, len(os.listdir(class_CDM_dir)))
        if len(os.listdir(class_CDM_dir)):
            save_npy_path = os.path.join(save_features_dir, pid+'_cell_density_haralick_features_'+class_name+'.npy')
            result = get_cell_density_haralick(class_CDM_dir)
            np.save(save_npy_path, np.array(result))

    haralick_labels = ["AngularSecondMoment",
                       "Contrast",
                       "Correlation",
                       "SumofSquares",
                       "InverseDifferenceMoment",
                       "SumAverage",
                       "SumVariance",
                       "SumEntropy",
                       "Entropy",
                       "DifferenceVariance",
                       "DifferenceEntropy",
                       "InformationMeasureofCorrelation1",
                       "InformationMeasureofCorrelation2"]
    statistic_list = ["mean", "median", "var", "kurt", "skew"]
    new_col = []
    for classname in class_name_list:
        for a in range(2):
            for statistic in statistic_list:
                for haralick_name in haralick_labels:
                    if a==0:
                        new_col.append(classname+ '_cell_density_' +haralick_name + '_' + statistic + '_HV')
                    else:
                        new_col.append(classname+ '_cell_density_' +haralick_name + '_' + statistic + '_D')
    new_col = ['pid'] + new_col
    haralick_df = pd.DataFrame(columns=new_col)
    haralick_df = haralick_df.append([{'pid': pid}], ignore_index=True)

    for classname in class_name_list:
        haralick_npy_path = os.path.join(save_features_dir, pid+'_cell_density_haralick_features_'+classname+'.npy')
        haralick_array = np.load(haralick_npy_path)
        add_label = []
        add_array = []
        for a in range(haralick_array.shape[0]):
            for b in range(haralick_array.shape[1]):
                for c in range(haralick_array.shape[2]):
                    if a==0:
                        add_label.append(classname + '_cell_density_' + haralick_labels[c] + '_' + statistic_list[b] + '_HV')
                    else:
                        add_label.append(classname + '_cell_density_' + haralick_labels[c] + '_' + statistic_list[b] + '_D')
                    add_array.append(haralick_array[a][b][c])
        haralick_df.loc[haralick_df['pid'] == pid, add_label] = add_array

    save_csv_path = os.path.join(save_features_dir, pid+'_cell_density_haralick_features_'+'_'.join(class_name_list)+'.csv')
    haralick_df.to_csv(save_csv_path, index=0)

def get_cell_density_features_TUM(slide_path, pid_dir):
    class_name_list =['TUM']
    pid = os.path.basename(slide_path).split('.')[0]
    CDM_dir = os.path.join(pid_dir, 'patch_2000_seg_array')
    save_features_dir = os.path.join(pid_dir, 'features')

    makedir(save_features_dir)
    for class_name in class_name_list:
        class_CDM_dir = os.path.join(CDM_dir, class_name)
        #print(pid, len(os.listdir(class_CDM_dir)))
        if len(os.listdir(class_CDM_dir)):
            save_npy_path = os.path.join(save_features_dir, pid+'_cell_density_haralick_features_'+class_name+'.npy')
            result = get_cell_density_haralick(class_CDM_dir)
            np.save(save_npy_path, np.array(result))
    
    haralick_labels = ["AngularSecondMoment",
                       "Contrast",
                       "Correlation",
                       "SumofSquares",
                       "InverseDifferenceMoment",
                       "SumAverage",
                       "SumVariance",
                       "SumEntropy",
                       "Entropy",
                       "DifferenceVariance",
                       "DifferenceEntropy",
                       "InformationMeasureofCorrelation1",
                       "InformationMeasureofCorrelation2"]
    statistic_list = ["mean", "median", "var", "kurt", "skew"]
    new_col = []
    for classname in class_name_list:
        for a in range(2):
            for statistic in statistic_list:
                for haralick_name in haralick_labels:
                    if a==0:
                        new_col.append(classname+ '_cell_density_' +haralick_name + '_' + statistic + '_HV')
                    else:
                        new_col.append(classname+ '_cell_density_' +haralick_name + '_' + statistic + '_D')
    new_col = ['pid'] + new_col
    haralick_df = pd.DataFrame(columns=new_col)
    haralick_df = haralick_df.append([{'pid': pid}], ignore_index=True)

    for classname in class_name_list:
        haralick_npy_path = os.path.join(save_features_dir, pid+'_cell_density_haralick_features_'+classname+'.npy')
        haralick_array = np.load(haralick_npy_path)
        add_label = []
        add_array = []
        for a in range(haralick_array.shape[0]):
            for b in range(haralick_array.shape[1]):
                for c in range(haralick_array.shape[2]):
                    if a==0:
                        add_label.append(classname + '_cell_density_' + haralick_labels[c] + '_' + statistic_list[b] + '_HV')
                    else:
                        add_label.append(classname + '_cell_density_' + haralick_labels[c] + '_' + statistic_list[b] + '_D')
                    add_array.append(haralick_array[a][b][c])
        haralick_df.loc[haralick_df['pid'] == pid, add_label] = add_array

    save_csv_path = os.path.join(save_features_dir, pid+'_cell_density_haralick_features_'+'_'.join(class_name_list)+'.csv')
    haralick_df.to_csv(save_csv_path, index=0)

def get_texture_features_TUM(slide_path, pid_dir):
    class_name_list =['TUM']
    #slide_path = '/home/linsy/ATPPP/data/slide/TCGA-2Y-A9H3-01Z.svs'
    pid = os.path.basename(slide_path).split('.')[0]
    #pid_dir = os.path.join('./data', pid)
    patch_dir = os.path.join(pid_dir, 'patch_2000')
    save_features_dir = os.path.join(pid_dir, 'features')

    makedir(save_features_dir)
    for class_name in class_name_list:
        class_patch_dir = os.path.join(patch_dir, class_name)
        #print(pid, len(os.listdir(class_patch_dir)))
        if len(os.listdir(class_patch_dir)):
            save_npy_path = os.path.join(save_features_dir, pid+'_texture_haralick_features_'+class_name+'.npy')
            result = get_texture_haralick(class_patch_dir)
            np.save(save_npy_path, np.array(result))

    haralick_labels = ["AngularSecondMoment",
                       "Contrast",
                       "Correlation",
                       "SumofSquares",
                       "InverseDifferenceMoment",
                       "SumAverage",
                       "SumVariance",
                       "SumEntropy",
                       "Entropy",
                       "DifferenceVariance",
                       "DifferenceEntropy",
                       "InformationMeasureofCorrelation1",
                       "InformationMeasureofCorrelation2"]
    statistic_list = ["mean", "median", "var", "kurt", "skew"]
    new_col = []
    for classname in class_name_list:
        for a in range(2):
            for statistic in statistic_list:
                for haralick_name in haralick_labels:
                    if a==0:
                        new_col.append(classname+ '_texture_' +haralick_name + '_' + statistic + '_HV')
                    else:
                        new_col.append(classname+ '_texture_' +haralick_name + '_' + statistic + '_D')
    new_col = ['pid'] + new_col
    haralick_df = pd.DataFrame(columns=new_col)
    haralick_df = haralick_df.append([{'pid': pid}], ignore_index=True)

    for classname in class_name_list:
        haralick_npy_path = os.path.join(save_features_dir, pid+'_texture_haralick_features_'+classname+'.npy')
        haralick_array = np.load(haralick_npy_path)
        add_label = []
        add_array = []
        for a in range(haralick_array.shape[0]):
            for b in range(haralick_array.shape[1]):
                for c in range(haralick_array.shape[2]):
                    if a==0:
                        add_label.append(classname + '_texture_' + haralick_labels[c] + '_' + statistic_list[b] + '_HV')
                    else:
                        add_label.append(classname + '_texture_' + haralick_labels[c] + '_' + statistic_list[b] + '_D')
                    add_array.append(haralick_array[a][b][c])
        haralick_df.loc[haralick_df['pid'] == pid, add_label] = add_array

    save_csv_path = os.path.join(save_features_dir, pid+'_texture_haralick_features_'+'_'.join(class_name_list)+'.csv')
    haralick_df.to_csv(save_csv_path, index=0)

def get_texture_features_DR_ICI(slide_path, pid_dir):
    class_name_list =['DR', 'ICI']
    #slide_path = '/home/linsy/ATPPP/data/slide/TCGA-2Y-A9H3-01Z.svs'
    pid = os.path.basename(slide_path).split('.')[0]
    #pid_dir = os.path.join('./data', pid)
    patch_dir = os.path.join(pid_dir, 'patch_500')
    save_features_dir = os.path.join(pid_dir, 'features')

    makedir(save_features_dir)
    for class_name in class_name_list:
        class_patch_dir = os.path.join(patch_dir, class_name)
        #print(pid, len(os.listdir(class_patch_dir)))
        if len(os.listdir(class_patch_dir)):
            save_npy_path = os.path.join(save_features_dir, pid+'_texture_haralick_features_'+class_name+'.npy')
            result = get_texture_haralick(class_patch_dir)
            np.save(save_npy_path, np.array(result))

    haralick_labels = ["AngularSecondMoment",
                       "Contrast",
                       "Correlation",
                       "SumofSquares",
                       "InverseDifferenceMoment",
                       "SumAverage",
                       "SumVariance",
                       "SumEntropy",
                       "Entropy",
                       "DifferenceVariance",
                       "DifferenceEntropy",
                       "InformationMeasureofCorrelation1",
                       "InformationMeasureofCorrelation2"]
    statistic_list = ["mean", "median", "var", "kurt", "skew"]
    new_col = []
    for classname in class_name_list:
        for a in range(2):
            for statistic in statistic_list:
                for haralick_name in haralick_labels:
                    if a==0:
                        new_col.append(classname+ '_texture_' +haralick_name + '_' + statistic + '_HV')
                    else:
                        new_col.append(classname+ '_texture_' +haralick_name + '_' + statistic + '_D')
    new_col = ['pid'] + new_col
    haralick_df = pd.DataFrame(columns=new_col)
    haralick_df = haralick_df.append([{'pid': pid}], ignore_index=True)

    for classname in class_name_list:
        haralick_npy_path = os.path.join(save_features_dir, pid+'_texture_haralick_features_'+classname+'.npy')
        haralick_array = np.load(haralick_npy_path)
        add_label = []
        add_array = []
        for a in range(haralick_array.shape[0]):
            for b in range(haralick_array.shape[1]):
                for c in range(haralick_array.shape[2]):
                    if a==0:
                        add_label.append(classname + '_texture_' + haralick_labels[c] + '_' + statistic_list[b] + '_HV')
                    else:
                        add_label.append(classname + '_texture_' + haralick_labels[c] + '_' + statistic_list[b] + '_D')
                    add_array.append(haralick_array[a][b][c])
        haralick_df.loc[haralick_df['pid'] == pid, add_label] = add_array

    save_csv_path = os.path.join(save_features_dir, pid+'_texture_haralick_features_'+'_'.join(class_name_list)+'.csv')
    haralick_df.to_csv(save_csv_path, index=0) 


def get_ICI_cell_count(slide_path, pid_dir):
    pid = os.path.basename(slide_path).split('.')[0]
    save_features_dir = os.path.join(pid_dir, 'features')

    class_name = 'ICI'
    save_path = os.path.join(save_features_dir, pid+'_cell_density_first_order_features_'+class_name+'.csv')
    class_CDM_dir = os.path.join(pid_dir, 'patch_500_seg_array', class_name)

    pids_result = []
    col1 = [ 'mean','median','var','skew','kurt']
    col2 = ['density','uniformity']
    col = ['pid']
    for col2name in col2:
        for col1name in col1:
            col.append(class_name+'_'+col2name+'_'+col1name)

    cell_density_list = []
    cell_uniformity_list = []
    result = []
    for patch_name in os.listdir(class_CDM_dir):
        patch_path = os.path.join(class_CDM_dir, patch_name)
        patch_array = np.load(patch_path)
        cell_density = np.sum(patch_array)
        cell_density_list.append(cell_density)
        cell_uniformity_list.append(np.std(patch_array))
    result.append(pid)
    result.append(np.mean(cell_density_list))
    result.append(np.median(cell_density_list))
    result.append(np.var(cell_density_list))
    result.append(pd.Series(cell_density_list).skew())
    result.append(pd.Series(cell_density_list).kurt())

    result.append(np.mean(cell_uniformity_list))
    result.append(np.median(cell_uniformity_list))
    result.append(np.var(cell_uniformity_list))
    result.append(pd.Series(cell_uniformity_list).skew())
    result.append(pd.Series(cell_uniformity_list).kurt())
    pids_result.append(result)
    df = pd.DataFrame(pids_result, columns=col)
    df.to_csv(save_path, index=0)

def get_TUM_cell_count(slide_path, pid_dir):
    pid = os.path.basename(slide_path).split('.')[0]
    save_features_dir = os.path.join(pid_dir, 'features')
    class_name = 'TUM'
    save_path = os.path.join(save_features_dir, pid+'_cell_density_first_order_features_'+class_name+'.csv')
    class_CDM_dir = os.path.join(pid_dir, 'patch_2000_seg_array', class_name)

    pids_result = []
    col1 = [ 'mean','median','var','skew','kurt']
    col2 = ['density','uniformity']
    col = ['pid']
    for col2name in col2:
        for col1name in col1:
            col.append(class_name+'_'+col2name+'_'+col1name)

    cell_density_list = []
    cell_uniformity_list = []
    result = []
    for patch_name in os.listdir(class_CDM_dir):
        patch_path = os.path.join(class_CDM_dir, patch_name)
        patch_array = np.load(patch_path)
        cell_density = np.sum(patch_array)
        cell_density_list.append(cell_density)
        cell_uniformity_list.append(np.std(patch_array))
    result.append(pid)
    result.append(np.mean(cell_density_list))
    result.append(np.median(cell_density_list))
    result.append(np.var(cell_density_list))
    result.append(pd.Series(cell_density_list).skew())
    result.append(pd.Series(cell_density_list).kurt())

    result.append(np.mean(cell_uniformity_list))
    result.append(np.median(cell_uniformity_list))
    result.append(np.var(cell_uniformity_list))
    result.append(pd.Series(cell_uniformity_list).skew())
    result.append(pd.Series(cell_uniformity_list).kurt())
    pids_result.append(result)
    df = pd.DataFrame(pids_result, columns=col)
    df.to_csv(save_path, index=0)

def separate_array(arr,classname):
    classname_list = ['DVLE', 'MVI', 'TUM','DR', 'ICI', 'HS']
    color = [[0, 255, 0], [0, 0, 255], [0, 150, 130], [255, 255, 170], [127, 85, 85], [0, 85, 0]]
    color_sum = []
    for a in color:
        color_sum.append(a[0]+a[1]+2*a[2])
    classname2rgb = dict(zip(classname_list,color_sum))
    r_img, g_img, b_img = arr[:, :, 0].copy(), arr[:, :, 1].copy(), arr[:, :, 2].copy()
    img = r_img +  g_img + 2*b_img
    dst = np.where(img == classname2rgb[classname]%256, 1, 0)
    #cv2.imwrite(os.path.join(save_dir,filepath.split('/')[-1].split('.')[0]+'_'+classname+'.png'), dst*255)
    return (dst*255).astype('uint8').T

def get_HS_count(slide_path, pid_dir, seg_result_dir):
    tile_size = 256
    overlap_size = 64
    pid = os.path.basename(slide_path).split('.')[0]
    foreground_dir = os.path.join(pid_dir, 'foreground')
    save_features_dir = os.path.join(pid_dir, 'features')
    save_path = os.path.join(save_features_dir, pid+'_density_first_order_features_HS.csv')

    col1 = ['mean', 'median', 'var', 'skew', 'kurt']
    col2 = ['density']
    col = ['pid']
    for col2name in col2:
        for col1name in col1:
            col.append('HS_' + col2name + '_' + col1name)
    pids_result = []
    tissue_path = os.path.join(foreground_dir, pid+'_L2.png')
    save_L1_path = os.path.join(seg_result_dir, pid+'_L1.png')
    target_mask = cv2.imread(tissue_path, cv2.IMREAD_GRAYSCALE)
    save_img = cv2.imread(save_L1_path)

    strided_mask = np.ones_like(target_mask)
    ones_mask = np.zeros_like(target_mask)
    factor = tile_size - overlap_size
    ones_mask[::factor, ::factor] = strided_mask[::factor, ::factor]
    strided_mask = ones_mask * target_mask
    X_idcs, Y_idcs = np.where(strided_mask)
    idcs_num = len(X_idcs)
    for idx in range(idcs_num):
        x_coord, y_coord = X_idcs[idx], Y_idcs[idx]
        if y_coord - factor >= 0:
            strided_mask[x_coord, y_coord - factor] = 1
        if y_coord + factor <= strided_mask.shape[1]:
            strided_mask[x_coord, y_coord + factor] = 1
        if x_coord - factor >= 0:
            strided_mask[x_coord - factor, y_coord] = 1
        if x_coord + factor <= strided_mask.shape[0]:
            strided_mask[x_coord + factor, y_coord] = 1
    X_idcs, Y_idcs = np.where(strided_mask)
    idcs_num = len(X_idcs)
    #print('Idcs_num %d' % (idcs_num), pid)
    result = []
    length_list = []
    for idx in range(idcs_num):
        x_coord, y_coord = X_idcs[idx], Y_idcs[idx]
        x_max_dim, y_max_dim = target_mask.shape
        # x = int(x_coord * self._resolution)
        # y = int(y_coord * self._resolution)
        x = int(x_coord)
        y = int(y_coord)

        # If Image goes out of bounds
        x = max(0, min(x, x_max_dim - tile_size))
        y = max(0, min(y, y_max_dim - tile_size))
        sub_img = save_img[x:x+tile_size, y:y+tile_size]
        mask = separate_array(sub_img, 'HS')
        contours, hireachy = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        length_list2 = []
        for i, contour in enumerate(contours):
            length = cv2.arcLength(contour, True)
            length_list2.append(length)
        length_list.append(np.sum(length_list2))
    result.append(pid)
    result.append(np.mean(length_list))
    result.append(np.median(length_list))
    result.append(np.var(length_list))
    result.append(pd.Series(length_list).skew())
    result.append(pd.Series(length_list).kurt())
    pids_result.append(result)
    df = pd.DataFrame(pids_result, columns=col)
    df.to_csv(save_path, index=0)
            #cv2.imwrite(os.path.join(save_dir, str(x)+'_'+str(y)+'.png'),sub_img)
            # Converting pil image to np array transposes the w and h

def get_area_length(contours,class_name,pid,res):
    area_list = []
    length_list = []
    eccentricity_list = []
    circularity_list = []
    convexity_list = []
    hu_moments_list = []
    n=0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if len(contour) < 5:
            continue
        if area<=0:
            continue
        n+=1
        length = cv2.arcLength(contour, True)
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments)
        (x, y), (minorAxisLength, majorAxisLength), angle = cv2.fitEllipse(contour)
        ellipse = cv2.fitEllipse(contour)
        # semi-major and semi-minor
        a = majorAxisLength / 2
        b = minorAxisLength / 2
        # Formula of eccentricity is :
        eccentricity = round(np.sqrt(pow(a, 2) - pow(b, 2)) / a, 2)
        circularity = pow(length,2)/area
        hull = cv2.convexHull(contour)
        convexity = cv2.contourArea(hull)/area

        area_list.append(area)
        length_list.append(length)
        eccentricity_list.append(eccentricity)
        circularity_list.append(circularity)
        convexity_list.append(convexity)
        hu_moments_list.append(hu_moments)
    

    if len(area_list)==0 or len(length_list)==0:
        area_list.append(0)
        length_list.append(0)
        eccentricity_list.append(0)
        circularity_list.append(0)
        convexity_list.append(0)

    area_list = [a_ for a_ in area_list if a_ == a_]
    length_list = [a_ for a_ in length_list if a_ == a_]
    eccentricity_list = [a_ for a_ in eccentricity_list if a_ == a_]
    circularity_list = [a_ for a_ in circularity_list if a_ == a_]
    convexity_list = [a_ for a_ in convexity_list if a_ == a_]

    area_s = pd.Series(area_list)
    length_s = pd.Series(area_list)
    eccentricity_s = pd.Series(eccentricity_list)
    circularity_s = pd.Series(circularity_list)
    convexity_s = pd.Series(convexity_list)

    res.loc[res['pid'] == pid, [class_name + '_area_sum']] = np.sum(area_list)
    res.loc[res['pid'] == pid, [class_name + '_area_mean']] = np.mean(area_list)
    res.loc[res['pid'] == pid, [class_name + '_area_median']] = np.median(area_list)
    res.loc[res['pid'] == pid, [class_name + '_area_var']] = np.var(area_list)
    res.loc[res['pid'] == pid, [class_name + '_area_max']] = np.max(area_list)
    res.loc[res['pid'] == pid, [class_name + '_area_skew']] = stats.skew(area_list)
    res.loc[res['pid'] == pid, [class_name + '_area_kurt']] = stats.kurtosis(area_list)

    res.loc[res['pid'] == pid, [class_name + '_length_sum']] = np.sum(length_list)
    res.loc[res['pid'] == pid, [class_name + '_length_mean']] = np.mean(length_list)
    res.loc[res['pid'] == pid, [class_name + '_length_median']] = np.median(length_list)
    res.loc[res['pid'] == pid, [class_name + '_length_var']] = np.var(length_list)
    res.loc[res['pid'] == pid, [class_name + '_length_max']] = np.max(length_list)
    res.loc[res['pid'] == pid, [class_name + '_length_skew']] = stats.skew(length_list)
    res.loc[res['pid'] == pid, [class_name + '_length_kurt']] = stats.kurtosis(length_list)

    res.loc[res['pid'] == pid, [class_name + '_eccentricity_sum']] = np.sum(eccentricity_list)
    res.loc[res['pid'] == pid, [class_name + '_eccentricity_mean']] = np.mean(eccentricity_list)
    res.loc[res['pid'] == pid, [class_name + '_eccentricity_median']] = np.median(eccentricity_list)
    res.loc[res['pid'] == pid, [class_name + '_eccentricity_var']] = np.var(eccentricity_list)
    res.loc[res['pid'] == pid, [class_name + '_eccentricity_max']] = np.max(eccentricity_list)
    res.loc[res['pid'] == pid, [class_name + '_eccentricity_skew']] = stats.skew(eccentricity_list)
    res.loc[res['pid'] == pid, [class_name + '_eccentricity_kurt']] = stats.kurtosis(eccentricity_list)

    res.loc[res['pid'] == pid, [class_name + '_circularity_sum']] = np.sum(circularity_list)
    res.loc[res['pid'] == pid, [class_name + '_circularity_mean']] = np.mean(circularity_list)
    res.loc[res['pid'] == pid, [class_name + '_circularity_median']] = np.median(circularity_list)
    res.loc[res['pid'] == pid, [class_name + '_circularity_var']] = np.var(circularity_list)
    res.loc[res['pid'] == pid, [class_name + '_circularity_max']] = np.max(circularity_list)
    res.loc[res['pid'] == pid, [class_name + '_circularity_skew']] = stats.skew(circularity_list)
    res.loc[res['pid'] == pid, [class_name + '_circularity_kurt']] = stats.kurtosis(circularity_list)

    res.loc[res['pid'] == pid, [class_name + '_convexity_sum']] = np.sum(convexity_list)
    res.loc[res['pid'] == pid, [class_name + '_convexity_mean']] = np.mean(convexity_list)
    res.loc[res['pid'] == pid, [class_name + '_convexity_median']] = np.median(convexity_list)
    res.loc[res['pid'] == pid, [class_name + '_convexity_var']] = np.var(convexity_list)
    res.loc[res['pid'] == pid, [class_name + '_convexity_max']] = np.max(convexity_list)
    res.loc[res['pid'] == pid, [class_name + '_convexity_skew']] = stats.skew(convexity_list)
    res.loc[res['pid'] == pid, [class_name + '_convexity_kurt']] = stats.kurtosis(convexity_list)
    return res

def get_shape_features(slide_path, pid_dir, seg_result_dir):
    pid = os.path.basename(slide_path).split('.')[0]
    save_features_dir = os.path.join(pid_dir, 'features')
    save_path = os.path.join(save_features_dir, pid+'_shap_features_DVLE_MVI_TUM.csv')

    classname_list = ['DVLE', 'MVI', 'TUM'] 
    columns = []
    columns.append('pid')
    columns2 = []
    for class_name in classname_list:
        for name1 in ['area', 'length', 'eccentricity', 'circularity', 'convexity']:
            for name2 in ['sum', 'mean', 'median', 'var', 'max', 'skew', 'kurt']:
                columns.append(class_name + '_' + name1 + '_' + name2)
                columns2.append(class_name + '_' + name1 + '_' + name2)
    res = pd.DataFrame(columns=columns)
    res = res.append([{'pid': pid}], ignore_index=True)

    save_L2_path = os.path.join(seg_result_dir, pid + '_L2.png')
    save_L1_path = os.path.join(seg_result_dir, pid + '_L1.png')

    for class_name in classname_list:
        if class_name in ['DVLE', 'MVI', 'TUM']:
            masks = separate_png(save_L2_path,class_name)
        else:
            masks = separate_png(save_L1_path, class_name)
        contours, hireachy = cv2.findContours(masks.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res = get_area_length(contours, class_name, pid, res)
    res.to_csv(save_path, index=False)

def get_features_table(slide_path, pid_dir, save_csv_path):
    pid = os.path.basename(slide_path).split('.')[0]
    pid_features_dir = os.path.join(pid_dir, 'features')

    shap_features_path = os.path.join(pid_features_dir, pid+'_shap_features_DVLE_MVI_TUM.csv')
    texture_features_DR_ICI_path = os.path.join(pid_features_dir, pid+'_texture_haralick_features_DR_ICI.csv')
    texture_features_TUM_path = os.path.join(pid_features_dir, pid+'_texture_haralick_features_TUM.csv')
    cell_density_FO_features_ICI_path = os.path.join(pid_features_dir, pid+'_cell_density_first_order_features_ICI.csv')
    cell_density_FO_features_HS_path = os.path.join(pid_features_dir, pid+'_density_first_order_features_HS.csv')
    cell_density_FO_features_TUM_path = os.path.join(pid_features_dir, pid+'_cell_density_first_order_features_TUM.csv')
    cell_density_HARA_features_TUM_path = os.path.join(pid_features_dir, pid+'_cell_density_haralick_features_TUM.csv')
    cell_density_HARA_features_ICI_path = os.path.join(pid_features_dir, pid+'_cell_density_haralick_features_ICI.csv')
    csv_path_list = [shap_features_path, texture_features_TUM_path, texture_features_DR_ICI_path, cell_density_FO_features_TUM_path, 
                     cell_density_FO_features_ICI_path, cell_density_FO_features_HS_path, cell_density_HARA_features_TUM_path, 
                     cell_density_HARA_features_ICI_path]
    pd_list = []
    for csv_path in csv_path_list:
        csv_df = pd.read_csv(csv_path)
        if csv_path==shap_features_path:
            pd_list.append(csv_df)
        else:
            pd_list.append(csv_df.iloc[:,1:])

    pid_result_df = pd.concat(pd_list, axis=1)
    if not os.path.exists(save_csv_path):
        pid_result_df.to_csv(save_csv_path, index=False)
    else:
        result_df = pd.read_csv(save_csv_path)
        result_df.loc[len(result_df.index)] = pid_result_df.loc[0,:].tolist()
        result_df.to_csv(save_csv_path, index=False)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("--slide_dir", type=str, help="the path of the folder of input WSIs", default="./data/slide")
    parse.add_argument("--seg_results_dir", type=str, help="the path of the folder of segmentation result", default="./data/seg_result")
    parse.add_argument("--data_dir", type=str, help="the path of the folder of intermediate data", default="./data/intermediate_data")
    parse.add_argument("--save_csv_path", type=str, help="the path of the folder of saving WSI features", default="./data/image_results.csv")
    args = parse.parse_args()

    for filename in os.listdir(args.slide_dir):
        slide_path = os.path.join(args.slide_dir, filename)
        pid = os.path.basename(slide_path).split('.')[0]
        pid_dir = os.path.join(args.data_dir, pid)
        pid_seg_result_dir = os.path.join(args.seg_results_dir, pid)
        makedir(pid_dir)
        print("Generating patches from the DenseUnet segmentation result...")
        get_patch_500(slide_path, pid_dir, pid_seg_result_dir)
        get_patch_2000(slide_path, pid_dir, pid_seg_result_dir)

        print("Segementing cells in patches..")
        cell_seg_patch_500(slide_path, pid_dir)
        cell_seg_patch_2000(slide_path, pid_dir)

        print("Calculating image features ..")
        get_cell_density_features_ICI(slide_path, pid_dir)
        get_cell_density_features_TUM(slide_path, pid_dir) 
        get_texture_features_DR_ICI(slide_path, pid_dir)
        get_texture_features_TUM(slide_path, pid_dir)
        get_TUM_cell_count(slide_path, pid_dir)
        get_ICI_cell_count(slide_path, pid_dir)
        get_HS_count(slide_path, pid_dir, pid_seg_result_dir)
        get_shape_features(slide_path, pid_dir, pid_seg_result_dir)
        get_features_table(slide_path, pid_dir, args.save_csv_path)

#['DVLE', 'MVI', 'TUM','DR', 'ICI', 'HS']