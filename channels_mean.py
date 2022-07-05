import numpy as np
import glob
import os
import tqdm


# 对semanticPOSS数据集的预处理，计算所有点云channels维的均值和标准差
# 注意数据集的文件系统要与skitti和sposs保持一致，root_folder/sequences/00/velodyne/*.bin
def compute_channels_mean_and_std(root_folder):
    lidars = glob.glob(root_folder + "/sequences/*/velodyne/*.bin")
    lidars.sort()
    means = []
    stds = []
    for lidar in lidars:
        points = np.fromfile(lidar, dtype=np.float32).reshape((-1, 4))
        distance = np.linalg.norm(points[:, 0:-1], ord=2, axis=1, keepdims=False)  # 得到每个点的距离
        # 计算每个点云文件5个channel的均值，注意intensity(反射强度)那一维要归一化(/255)
        range_mean = distance.mean()
        x_mean = points[:, 0].mean()
        y_mean = points[:, 1].mean()
        z_mean = points[:, 2].mean()
        i_mean = (points[:, 3] / 255.0).mean()
        range_std = distance.std()
        x_std = points[:, 0].std()
        y_std = points[:, 1].std()
        z_std = points[:, 2].std()
        i_std = (points[:, 3] / 255.0).std()
        # 顺序为range,x,y,z,remission
        means.append([range_mean, x_mean, y_mean, z_mean, i_mean])
        stds.append([range_std, x_std, y_std, z_std, i_std])
    means_np = np.array(means)
    stds_np = np.array(stds)
    # 再将所有点云文件的channels均值和标准差再求一遍均值
    means_total = means_np.mean(axis=0)
    stds_total = stds_np.mean(axis=0)
    print(f"channel_means: {means_total}\nchannels_std: {stds_total}")
    return means_total, stds_total

# 统计各类点云所占比例，同样，数据的文件系统格式要参照skitti或sposs
def content_statistic(root_folder, num_classes = 5):
    label_files = glob.glob(root_folder + "/sequences/*/labels/*.label")
    label_files.sort()
    total_count = 0
    class_count = [0 for i in range(num_classes)]
    for label_file in label_files:
        label = np.fromfile(label_file, dtype=np.int32)
        total_count += len(label)  # 总点云数量
        # 每类点云的数量，这里假定label是从0开始顺序的数字编码
        for i in range(num_classes):
            class_count[i] += len(label[label == i])
    print("content: ", np.array(class_count) / total_count)
    return np.array(class_count) / total_count


if __name__ == "__main__":
    print("\n-------------- start --------------\n")

    root_folder = "/home/cidi/semanticCIDI_COARSE"
    num_classes = 5
    content_statistic(root_folder, num_classes=5)
    compute_channels_mean_and_std(root_folder)

    print("\n--------------- end ---------------\n")
