'''
PLABEL代表semantic-segmentation-editor所输出的类别序号
SPOSS代表semanticPOSS的类别序号
SKITTI代表semanticKITTI的类别序号
SCIDI代表semanticCIDI的类别序号，我自己定义的
SCIDI_COARSE代表semanticCIDI的粗标签类别序号，我自己定义的
'''

# semantic segmentation editor输出的label序号，默认选择Cityscapes的标签
_PLABEL = {
    0: "VOID",
    1: "Road", 2: "Sidewalk", 3: "Parking", 4: "Rail Track", 5: "Person",
    6: "Rider", 7: "Car", 8: "Truck", 9: "Bus", 10: "On Rails",
    11: "Motorcycle", 12: "Bicycle", 13: "Caravan", 14: "Trailer", 15: "Building",
    16: "Wall", 17: "Fence", 18: "Guard Rail", 19: "Bridge", 20: "Tunnel",
    21: "Pole", 22: "Pole Group", 23: "Traffic Sign", 24: "Traffic Light", 25: "Vegetation",
    26: "Terrain", 27: "Sky", 28: "Ground", 29: "Dynamic", 30: "Static",
}
# cidi粗标签的序号对应，我自己定义的，后期可修改
# 6: living在editor中用24: Traffic Light表示，
# 7: manual在editor中用19: Bridge表示，
# 8: other_object在editor中用23: Traffic Sign表示，
# 14: traffic_object在editor中用21: Pole表示，
# 其余类别都有对应的名称
_SCIDI = {
    0: "unlabeled",  # 离散点、无法辨识的点
    1: "person",  # 行人、背包的人、打伞的人、拖着拉杆箱的人等，只要人带的东西比人小，都归入此类
    2: "rider",  # 自行车、电动车、摩托车、三轮车等，只要是敞篷的小型载具，都归入此类
    3: "car",  # 轿车、SUV、面包车、皮卡、叉车，只要是中型的有外壳的载具，都归入此类
    4: "bus",  # 公交车、中巴、大巴，只要是大型的有外壳的载具，都归入此类
    5: "truck",  # 大货车、半挂、泥罐车、油罐车、吊车，只要是大型的敞篷载具，都归入此类
    6: "living",  # 活物，如宠物、机器人等，能自主运动的物体，都归入此类

    # 手推车、平板车、垃圾车等(正在运动的，包含施力的人)
    # 与other_object不同，manual是由于人的施力，正在运动的物体，如人正在拉平板车
    7: "manual",

    # 手推车、平板车、垃圾车等(静止的，旁边没有人在施力)
    # 与manual不同，other_object是静止的物体，如停在路旁的垃圾车
    8: "other_object",

    9: "vegetation",  # 地表植被、冬青绿化带、行道树等所有植被，
    10: "building",  # 建筑，包括外墙、头顶高架桥、天桥、横跨道路的大广告牌等，人造的又高又大的，都归入此类
    11: "fence",  # 围栏、施工护栏、有可能出现在行车道上
    12: "road",  # 行车道，就是车能跑的路(物理意义上的能跑)
    13: "sidewalk",  # 自行车道和人行道
    14: "traffic_object",  # 其他交通物品，比如路灯、电线杆、交通标志、信号灯等
}

_SCIDI_COARSE = {
    0: "unlabeled",  # 离散点、无法辨识的点
    1: "person",  # 马路上的小物件
    2: "car",  # 所有车
    3: "road",  # 可行驶(物理意义)的道路
    4: "vegetation",  # 背景(大部分是建筑和植被)
}

_PLABEL2SPOSS = {
    0: 0,
    1: 22, 2: 22, 3: 22, 4: 3, 5: 4,
    6: 6, 7: 7, 8: 8, 9: 3, 10: 3,
    11: 6, 12: 6, 13: 3, 14: 8, 15: 15,
    16: 15, 17: 17, 18: 17, 19: 22, 20: 18,
    21: 13, 22: 13, 23: 13, 24: 13, 25: 9,
    26: 9, 27: 18, 28: 22, 29: 1, 30: 18,
}

_PLABEL2SKITTI = {
    0: 0,
    1: 40, 2: 48, 3: 44, 4: 16, 5: 30,
    6: 31, 7: 10, 8: 18, 9: 13, 10: 16,
    11: 15, 12: 11, 13: 20, 14: 20, 15: 50,
    16: 50, 17: 51, 18: 99, 19: 40, 20: 40,
    21: 80, 22: 80, 23: 81, 24: 81, 25: 70,
    26: 72, 27: 1, 28: 40, 29: 0, 30: 0,
}

_PLABEL2SCIDI = {
    0: 0,
    1: 8, 2: 8, 3: 8, 4: 0, 5: 1,
    6: 2, 7: 3, 8: 4, 9: 10, 10: 0,
    11: 2, 12: 2, 13: 0, 14: 0, 15: 6,
    16: 6, 17: 7, 18: 7, 19: 8, 20: 8,
    21: 9, 22: 9, 23: 9, 24: 9, 25: 5,
    26: 5, 27: 0, 28: 8, 29: 0, 30: 0,
}

_PLABEL2SCIDI_COARSE = {
    0: 0, 1: 3, 5: 1, 7: 2, 25: 4,
    2: 0, 3: 0, 4: 0, 6: 0, 8: 0, 9: 0, 10: 0,
    11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0,
    17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0,
    23: 0, 24: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0,
}

_SPOSS2PLABEL = {
    0: 0, 1: 0, 2: 0, 3: 0,
    4: 5, 5: 5, 6: 6, 7: 7, 8: 8,
    9: 25, 10: 0, 11: 0, 12: 0, 13: 21,
    14: 0, 15: 15, 16: 0, 17: 17, 18: 0,
    19: 0, 20: 0, 21: 12, 22: 1
}

_SKITTI2PLABEL = {
    0: 0, 1: 0, 10: 7, 11: 12, 13: 9,
    15: 11, 16: 10, 18: 8, 20: 29, 30: 5,
    31: 6, 32: 6, 40: 1, 44: 3, 48: 2,
    49: 0, 50: 15, 51: 17, 52: 0, 60: 0,
    70: 25, 71: 21, 72: 26, 80: 21, 81: 23, 99: 22,
    252: 7, 253: 6, 254: 5, 255: 6, 256: 10, 257: 9, 258: 8, 259: 29
}

_SCIDI_COARSE2PLABEL = {
    0: 0, 1: 5, 2: 7, 3: 1, 4: 25,
}

_SPOSS2SCIDI_COARSE = {
    0: 0,  # unlabeled
    1: 1, 2: 1, 4: 1, 5: 1, 6: 1, 14: 1,  # person
    3: 3, 7: 2, 8: 2,  # car
    22: 3,  # road
    9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 15: 4, 16: 4, 17: 4, 18: 4, 19: 4, 20: 4, 21: 4  # vegetation
}

# TODO:kitti的数据对应到粗标签类
_SKITTI2SCIDI_COARSE = {
    # unlabeled
    # person
    # car
    # road
    # vegetation
}

# TODO:等最后的标签类别定下来再写这个字典
_SCIDI2PLABEL = {

}
