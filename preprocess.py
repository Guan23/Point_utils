# -*- coding:utf-8 -*-
# author: Guan
# @file: preprocess.py

import numpy as np
import glob
import os
import sys
import pypcd.pypcd as pypcd


# 速腾聚创的激光雷达pcd点云转bin文件，其pcd的data为binary，有五维，分别是x，y，z，intensity，ring
# 其中x，y，z均为numpy.int16类型，后两维为uint8类型
# 要使用到pypcd库来进行pcd文件的解析

# 速腾聚创的pcd点云文件转bin
def robosense2skitti(pcd_path):
    pc = pypcd.PointCloud.from_path(pcd_path)
    pointcloud = np.array(pc.pc_data.tolist(), dtype=np.float32)[:, :-1]  # 把numpy.void转为ndarray
    pointcloud[:, 3] /= 255  # intensity归一化
    pointcloud[:, 0:4] /= 100  # 速腾的xyz是乘100后按np.int16存储的，需要除回去
    pointcloud = np.reshape(pointcloud, (-1))
    return pointcloud


# cidi的livox激光点云数据存在(0,0,0)点，要删除，并且其最后一列反射强度均为0
def livox2skitti(lidar_path):
    pointcloud = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 4])
    i = 0
    while i < len(pointcloud):
        if pointcloud[i].any() == 0.0:
            pointcloud = np.delete(pointcloud, i, axis=0)
            i -= 1
        i += 1
    pointcloud[:, -1] = 0.0

    return pointcloud


# 北大的SemanticPOSS，最后一列的反射强度范围是0～255,要进行归一化
def sposs2skitti(lidar_path):
    pointcloud = np.fromfile(lidar_path, dtype=np.float32, count=-1).reshape([-1, 4])
    divider = np.ones_like(pointcloud)
    divider[:, -1] = 255.0
    pointcloud = np.divide(pointcloud, divider)

    return pointcloud


# 对一个文件夹内所有bin点云文件进行批处理
def batch_convert(lidar_folder, save_folder, ori_format="livox"):
    if ori_format != "robosense":
        lidars = glob.glob(lidar_folder + "/*.bin")
        lidars.sort()
        os.makedirs(save_folder, exist_ok=True)
        for i, lidar in enumerate(lidars):
            if ori_format == "livox":
                pointcloud = livox2skitti(lidar)
            elif ori_format == "sposs":
                pointcloud = sposs2skitti(lidar)
            else:
                print("暂不支持livox,sposs和robosense以外的其他格式")
                sys.exit(-1)
            pointcloud.tofile(os.path.join(save_folder, str(i).zfill(6) + ".bin"))
            print(os.path.join(save_folder, str(i).zfill(6) + ".bin"), " saved!")
    else:
        lidars = glob.glob(lidar_folder + "/*.pcd")
        lidars.sort()
        os.makedirs(save_folder, exist_ok=True)
        for i, lidar in enumerate(lidars):
            pointcloud = robosense2skitti(lidar)
            pointcloud.tofile(os.path.join(save_folder, str(i).zfill(6) + ".bin"))
            print(os.path.join(save_folder, str(i).zfill(6) + ".bin"), " saved!")


if __name__ == '__main__':
    print("\n-------------- start --------------\n")

    lidar_folder = "./bin"
    save_folder = "./convert"
    batch_convert(lidar_folder, save_folder, ori_format="livox")

    print("\n--------------- end ---------------\n")
