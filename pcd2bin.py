# _*_ encoding=utf-8 _*_
import pypcd.pypcd as pypcd
import numpy as np
import os
import glob
import sys
import shutil

import label_dict as LD


# 速腾聚创的激光雷达pcd点云转bin文件，其pcd的data为binary，有五维，分别是x，y，z，intensity，ring
# 其中x，y，z均为numpy.int16类型，后两维为uint8类型
# 要使用到pypcd库来进行pcd文件的解析

# 速腾聚创的pcd点云文件转bin
def pcd2bin(pcd_path):
    pc = pypcd.PointCloud.from_path(pcd_path)
    pc_np = np.array(pc.pc_data.tolist(), dtype=np.float32)[:, :-1]  # 把numpy.void转为ndarray
    pc_np[:, 3] /= 255  # intensity归一化
    pc_np[:, 0:4] /= 100  # 速腾的xyz是乘100后按np.int16存储的，需要除回去
    pc_np = np.reshape(pc_np, (-1))
    return pc_np


# 注释工具的label(.pcd格式)转训练的label(.bin格式)
def plabel2blabel(pcd_label):
    pc = pypcd.PointCloud.from_path(pcd_label)
    pc_np = np.array(pc.pc_data.tolist(), dtype=np.float32)
    bin_label = pc_np[:, 3].astype(np.int32)
    # 将注释工具的label序号映射到对应数据集
    for i in range(len(bin_label)):
        bin_label[i] = LD._PLABEL2SCIDI_COARSE[bin_label[i]]  # 这里选择转成对应的数据集
    return bin_label


# 批处理，需要pcd文件的文件夹和保存.bin文件的文件夹，ftype为label则是editor工具得到的标注pcd文件转bin，为pcd则是速腾聚创的pcd点云转bin点云
def batchprocess(pcd_folder, save_folder, ftype="label"):
    filepaths = glob.glob(pcd_folder + "/*.pcd")
    filepaths.sort()  # 文件名排序
    for i, path in enumerate(filepaths):
        if ftype == "label":
            pc_np = plabel2blabel(path)
            suffix = ".label"
        elif ftype == "pcd":
            pc_np = pcd2bin(path)
            suffix = ".bin"
        else:
            print("暂时只支持label和pcd类型的文件！")
            sys.exit(-1)
        # savefile = os.path.join(save_folder, str(i).zfill(6) + suffix)  # 保存的文件名按照Kitti的格式
        savefile = os.path.join(path.replace(".pcd", ".label").replace("pcd_labels","bin_labels"))  # 保存的文件名照原名
        pc_np.tofile(savefile)
        print("{} file saved!".format(savefile))


if __name__ == "__main__":
    print("\n-------------- start --------------\n")

    # pcd_label = "./pcd_label/1638582124.399924994.pcd"
    # plabel2blabel(pcd_label)

    pcd_folder = "./pcd_labels"

    save_folder = "./bin_labels"
    batchprocess(pcd_folder, save_folder)

    print("\n--------------- end ---------------\n")
