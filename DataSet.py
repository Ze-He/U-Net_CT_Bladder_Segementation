import os
import glob
import cv2
import numpy as np


filenames_train = ['chenjiating17', 'fenghuarong27', 'kama19', 'lixiuqing23',
                   'liyongfen22', 'pengqinglan24', 'pufuqiong14', 'renlongxian12',
                   'tangwenying28', 'wanggequn26', 'wangqiying20', 'wenzhubi15',
                   'wuhongmei29', 'xuezhihui18']
filenames_var = ['yanliangzhen30', 'yuminglan11', 'zhangxianqin13', 'zhaobinglan21']
filenames_test = ['zhengchengfang25', 'zhengqunying16']

source_folder = r'G:\2020\python\U_Net_CT_Bladder_Segementation\data_crop_reshape'

image_train = []
image_var = []
image_test = []
mask_train = []
mask_var = []
mask_test = []
imageTrain = []
imageVar = []
imageTest = []
maskTrain = []
maskVar = []
maskTest = []


def load_image():
    num = 0
    for i in range(len(filenames_train)):
        # images
        dir_name = os.path.join(source_folder, filenames_train[i])
        dir_name_image = os.path.join(dir_name, 'images')
        dir_image_png = glob.glob(dir_name_image + '/*.png')
        for j in range(len(dir_image_png)):
            image_train.append(cv2.imread(dir_image_png[j]))
            num = num + 1

    imageTrain = np.array(image_train, dtype=np.float32)

    num = 0
    for i in range(len(filenames_var)):
        # images
        dir_name = os.path.join(source_folder, filenames_var[i])
        dir_name_image = os.path.join(dir_name, 'images')
        dir_image_png = glob.glob(dir_name_image + '/*.png')
        for j in range(len(dir_image_png)):
            image_var.append(cv2.imread(dir_image_png[j]))
            num = num + 1

    imageVar = np.array(image_var, dtype=np.float32)

    num = 0
    for i in range(len(filenames_test)):
        # images
        dir_name = os.path.join(source_folder, filenames_test[i])
        dir_name_image = os.path.join(dir_name, 'images')
        dir_image_png = glob.glob(dir_name_image + '/*.png')
        for j in range(len(dir_image_png)):
            image_test.append(cv2.imread(dir_image_png[j]))
            num = num + 1

    imageTest = np.array(image_test, dtype=np.float32)

    return imageTrain, imageVar, imageTest


def load_mask():
    num = 0
    for i in range(len(filenames_train)):
        # masks
        dir_name = os.path.join(source_folder, filenames_train[i])
        dir_name_mask = os.path.join(dir_name, 'masks')
        dir_image_png = glob.glob(dir_name_mask + '/*.png')
        for j in range(len(dir_image_png)):
            mask_train.append(cv2.imread(dir_image_png[j]))
            num = num + 1

    maskTrain = np.array(mask_train, dtype=np.float32)

    num = 0
    for i in range(len(filenames_var)):
        # masks
        dir_name = os.path.join(source_folder, filenames_var[i])
        dir_name_mask = os.path.join(dir_name, 'masks')
        dir_image_png = glob.glob(dir_name_mask + '/*.png')
        for j in range(len(dir_image_png)):
            mask_var.append(cv2.imread(dir_image_png[j]))
            num = num + 1

    maskVar = np.array(mask_var, dtype=np.float32)

    num = 0
    for i in range(len(filenames_test)):
        # masks
        dir_name = os.path.join(source_folder, filenames_test[i])
        dir_name_mask = os.path.join(dir_name, 'masks')
        dir_image_png = glob.glob(dir_name_mask + '/*.png')
        for j in range(len(dir_image_png)):
            mask_test.append(cv2.imread(dir_image_png[j]))
            num = num + 1

    maskTest = np.array(mask_test, dtype=np.float32)

    return maskTrain, maskVar, maskTest
