import cv2
import os
import glob


filenames = ['chenjiating17', 'fenghuarong27', 'kama19', 'lixiuqing23',
             'liyongfen22', 'pengqinglan24', 'pufuqiong14', 'renlongxian12',
             'tangwenying28', 'wanggequn26', 'wangqiying20', 'wenzhubi15',
             'wuhongmei29', 'xuezhihui18', 'yanliangzhen30', 'yuminglan11',
             'zhangxianqin13', 'zhaobinglan21', 'zhengchengfang25', 'zhengqunying16']


def Crop(dir):
    img = cv2.imread(dir)
    reshape_row = img.shape[0]
    reshape_col = img.shape[1]
    a = int(reshape_row / 2 - 160)
    b = int(reshape_row / 2 + 90)
    c = int(reshape_col / 2 - 125)
    d = int(reshape_col / 2 + 125)

    img_crop = img[a:b, c:d, :]
    return img_crop


def Crop_image(source_folder, target_folder, ):
    # 读入20个项目
    for i in range(len(filenames)):
        # 读取images, masks
        dir_name = os.path.join(source_folder, filenames[i])
        dir_name_image = os.path.join(dir_name, 'images')
        dir_image_png = glob.glob(dir_name_image + '/*.png')
        dir_name_mask = os.path.join(dir_name, 'masks')
        dir_mask_png = glob.glob(dir_name_mask + '/*.png')
        # 写入images, masks
        dir_target = os.path.join(target_folder, filenames[i])
        if not os.path.exists(dir_target): os.makedirs(dir_target)
        dir_target_image = os.path.join(dir_target, 'images')
        if not os.path.exists(dir_target_image): os.makedirs(dir_target_image)
        dir_target_mask = os.path.join(dir_target, 'masks')
        if not os.path.exists(dir_target_mask): os.makedirs(dir_target_mask)

        for j in range(len(dir_image_png)):
            image_crop = Crop(dir_image_png[j])
            image_reshape = cv2.resize(image_crop, dsize=(572,572), interpolation=cv2.INTER_CUBIC)
            mask_crop = Crop(dir_mask_png[j])
            mask_reshape = cv2.resize(mask_crop, dsize=(572,572), interpolation=cv2.INTER_CUBIC)
            image_base = os.path.basename(dir_image_png[j])
            mask_base = os.path.basename(dir_mask_png[j])
            cv2.imwrite((dir_target_image +'\\'+ image_base), image_reshape)
            cv2.imwrite((dir_target_mask +'\\'+ mask_base), mask_reshape)


source_folder = r'G:\2020\python\Heze_CT_bladder_segementation\data'
target_folder = r'G:\2020\python\Heze_CT_bladder_segementation\data_crop_reshape'

Crop_image(source_folder, target_folder)

