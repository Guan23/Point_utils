# _*_ encoding=utf-8 _*_
import pypcd.pypcd as pypcd
import shutil
import numpy as np
import os
import glob
import label_dict as LD
import tqdm

# bin格式的label转成sseditor标注工具可以识别的pcd格式的label
def blabel2plabel(bin_path, label_path):
    bin_lidar = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    bin_label = np.fromfile(label_path, dtype=np.int32).reshape(-1)
    assert bin_lidar.shape[0] == bin_label.shape[0], "lidar and label shape not same!"
    for i in range(len(bin_label)):
        # 这里的类别不同数据集处理方式不同，sposs先全赋值0，然后再按照具体类别赋数字
        # 而skitti不知道先赋的是啥，但是只要是类别范围之外的数字均为unlabeled
        if bin_label[i] > 260 or bin_label[i] < -1:
            bin_label[i] = 0
        bin_label[i] = LD._SCIDI_COARSE2PLABEL[bin_label[i]]  # 这里更换字典
    bin_lidar[:, 3] = bin_label  # 把label填充到第四列
    # bin_lidar = np.insert(bin_lidar, 4, values=0, axis=1)  # 第五列object暂时留空
    # 存放路径
    PCD_FILE_PATH = label_path.replace(".label", ".pcd")  # 生成的文件名与原文件同名，只是后缀不同
    if os.path.exists(PCD_FILE_PATH):
        os.remove(PCD_FILE_PATH)
    # 写文件句柄
    handle = open(PCD_FILE_PATH, 'a')
    # 得到点云点数
    point_num = len(bin_lidar)
    # pcd头部，跟PCL(C++)库生成的<PCL::PointXYZL>类型保持一致的表头信息
    handle.write(
        '# .PCD v0.7 - Point Cloud Data file format\n'
        'VERSION 0.7\n'
        'FIELDS x y z label\n'
        'SIZE 4 4 4 4\n'
        'TYPE F F F U\n'
        'COUNT 1 1 1 1')
    handle.write('\nWIDTH ' + str(point_num))
    handle.write('\nHEIGHT 1')
    handle.write('\nVIEWPOINT 0 0 0 1 0 0 0')
    handle.write('\nPOINTS ' + str(point_num))
    handle.write('\nDATA ascii')
    # 依次写入点
    for point in bin_lidar:  # 这里我用了5列，分别如上面头文件所示，注意最后两列要取整
        string = '\n' + str(point[0]) + ' ' + str(point[1]) + ' ' + str(point[2]) + ' ' + str(int(point[3]))
        handle.write(string)
    handle.close()
    return PCD_FILE_PATH

# 批处理
def batch_process(data_folder, save_folder):
    bin_paths = glob.glob(data_folder+"/velodyne/*.bin")
    label_paths = glob.glob(data_folder+"/labels/*.label")
    bin_paths.sort()
    label_paths.sort()
    frames_num = len(bin_paths)
    for i in tqdm.tqdm(range(frames_num)):
        pcd_ori_path = blabel2plabel(bin_paths[i], label_paths[i])
        pcd_save_path = os.path.join(save_folder, pcd_ori_path.split("/")[-1])
        shutil.move(pcd_ori_path, pcd_save_path)


if __name__ == "__main__":
    print("\n-------------- start --------------\n")

    # bin_path = "./semanticKITTI/velodyne/000001.bin"
    # label_path = "./semanticKITTI/labels/000001.label"
    # blabel2plabel(bin_path, label_path)

    data_folder = "/home/cidi/Cylinder3D/demo_dir"
    save_folder = os.path.join(data_folder, "pcd_labels")
    os.makedirs(save_folder, exist_ok=True)
    batch_process(data_folder, save_folder)

    print("\n--------------- end ---------------\n")
