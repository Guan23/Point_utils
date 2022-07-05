import numpy as np
from pypcd import pypcd

def plabel2blabel(pcd_label):
    pc = pypcd.PointCloud.from_path(pcd_label)
    pc_np = np.array(pc.pc_data.tolist(), dtype=np.float32)
    bin_label = pc_np[:, 3].astype(np.int32)
    bin_label.tofile(pcd_label.replace(".pcd", ".bin"))

if __name__ == "__main__":
    print("\n-------------- start --------------\n")

    pcd_label = "./pcd_label/1638582124.399924994.pcd"
    pc = pypcd.PointCloud.from_path(pcd_label)
    pc_np = np.array(pc.pc_data.tolist(), dtype=np.float32)
    print(pc_np)
    bin_label = pc_np[:, 3].astype(np.int32)
    bin_label.tofile(pcd_label.replace(".pcd", ".bin"))
    print(bin_label.shape)
    print(bin_label)

    print("\n--------------- end ---------------\n")
