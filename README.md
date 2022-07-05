# Point_utils

依赖：pypcd（主要是用来读取robosense的pcd点云）

1、此工程涵盖了各种三维语义分割点云数据集的预处理，主要是不同数据集之间的格式转换。
主要格式有：semanticKITTI、semanticPOSS、livox_horizon、robosense32、sseditor（日立的那个标注工具）
这里定义两个基准，训练时，把数据格式都转成semanticKITTI的格式，标注时都转成sseditor的格式。

2、sposs、livox和robosense转skitti的代码都在preprocess.py里
sseditor标注工具使用的点云格式跟robosense的pcd是一致的，故只需要额外写sseditor和skitti之间的label互相转换的函数即可
sseditor的pcdlabel转skitti的binlabel在plabel_to_blabel.py中，而binlabel转pcdlabel在bin2pcd.py中

3、channels_mean.py是用来计算semanticPOSS数据集的channels的mean和std的，计算结果如下(只计算训练集，没有03sequence)：
img_means: # range,x,y,z,signal
      - 23.661396
      - 0.7192412
      - 1.6966875
      - -0.60604143
      - 0.05825233
img_stds: # range,x,y,z,signal
      - 18.95187
      - 18.496365
      - 23.658104
      - 1.7329011
      - 0.06518337
     
4、pointvisual.py是简单的点云可视化脚本
