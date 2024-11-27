"""
    已经发布到 CSDN ：https://blog.csdn.net/qq_27816785/article/details/138634496
    使用 laspy v2.3
    1. 注意：
        * 实测版本 laspy v2.3，2.x版本应该都可用，但不适合1.x版本。
        * las 存储数据时，需要设置 scales 和 offsets，否则会出现精度问题。
        * las 存储颜色时，数值类型为 16 位无符号整型。rgb = (normal_rgb * 65535).astype(np.uint16)

    2. las 格式原生支持的属性字段，与las版本密切相关。官方说明：https://laspy.readthedocs.io/en/latest/intro.html#point-records

    3. 对 scales 和 offsets 的理解：
        https://gis.stackexchange.com/questions/346679/if-converting-xyz-to-las-using-e-g-laspy-then-how-to-specify-offset-and-scal
        比例scales 表明数据的准确性。 0.001 是毫米精度。这意味着如果您的坐标是 0.123456，它将被限制为 0.123。
        偏移offset 的目的是避免整数溢出。假设您要存储 123456789.123。在 LAS 文件中，您实际上将存储一个整数：123456789123，
            该整数将在读取时使用比例因子转换为 123456789.123。但 123456789123 比 32 位整数所能存储的要大得多。
            因此存储时，将该值偏移 123450000，实际存的是6789123。(6789123 * 0.001 + 123450000 = 123456789.123)
            所以 xmin、ymax、zmin 是有效的选项，但我们通常更喜欢将这些值四舍五入到最接近的整数米/公里
"""
import warnings

import laspy
import numpy as np


def read_las_fit(filename, attrs=None):
    """
    读取 las 文件，获取三维坐标 xyz, 颜色 rgb, 属性 attr_dict。当文件没有 RGB 信息时，返回全0的 RGB 信息
    Args:
        filename: <str> las 文件路径
        attrs: <list> 需要额外获取的属性信息 如 ['label']

    Returns:
        xyz, rgb, attr_dict
    """
    if attrs is None:
        attrs = []

    # 默认返回 scales, offsets ，合并 ["scales", "offsets"]
    attrs = list(set(attrs + ["scales", "offsets"]))

    # 读取点云
    inFile = laspy.read(filename)
    # inFile.point_format.dimensions可以获取所有的维度信息
    N_points = len(inFile)
    x = np.reshape(inFile.x, (N_points, 1))
    y = np.reshape(inFile.y, (N_points, 1))
    z = np.reshape(inFile.z, (N_points, 1))
    xyz = np.hstack((x, y, z))
    ''' 注意。如果是大写的 X Y Z，需要转换后才是真实坐标: real_x = scale[0] * inFile.X + offset[0] '''

    # 初始化 rgb 全是 0
    rgb = np.zeros((N_points, 3), dtype=np.uint16)
    if hasattr(inFile, "red") and hasattr(inFile, "green") and hasattr(inFile, "blue"):
        r = np.reshape(inFile.red, (N_points, 1))
        g = np.reshape(inFile.green, (N_points, 1))
        b = np.reshape(inFile.blue, (N_points, 1))
        # i = np.reshape(inFile.Reflectance, (N_points, 1))
        rgb = np.hstack((r, g, b))
    else:
        print(f"注意：{filename.split('/')[-1]} 没有RGB信息，返回全0的RGB信息！")

    # 组织其他属性信息
    attr_dict = {}
    for attr in attrs:
        value = None
        # 先判断 是否为额外属性
        if hasattr(inFile, attr):
            value = getattr(inFile, attr)
        # 再判断 header 中是否有该属性
        elif hasattr(inFile.header, attr):
            value = getattr(inFile.header, attr)
        else:
            warnings.warn(f"{filename.split('/')[-1]} 没有属性= {attr} 的信息！", category=Warning)  # 使用 warnning 警告

        if hasattr(value, "array"):
            attr_dict[attr] = np.array(value)
        else:
            attr_dict[attr] = value
    return xyz, rgb, attr_dict


def write_las_fit(out_file, xyz, rgb=None, attrs=None):
    """
    将点云数据写入 las 文件，支持写入 坐标xyz, 颜色rgb, 属性attrs
    Args:
        out_file: 输出文件路径
        xyz: 点云坐标 ndarray (N, 3)
        rgb: 点云颜色 ndarray (N, 3)
        attrs:
            固有属性：file_source_id, gps_time, Intensity, Number of Returns, ....
            额外属性：label, pred, ...
            注意：如果不传入 scales 和 offsets，则会自动计算
    Returns:

    """
    if attrs is None:
        attrs = {}

    # 1. 创建 las 文件头。point_format和version决定了las支持哪些固有属性
    # 详情见 https://pylas.readthedocs.io/en/latest/intro.html?highlight=red#point-records
    header = laspy.LasHeader(point_format=7, version="1.4")  # 7 支持rgb

    # 自动计算 scales 和 offsets，确保坐标精度无损
    # https://stackoverflow.com/questions/77308057/conversion-accuracy-issues-of-e57-to-las-in-python-using-pye57-and-laspy
    if "offset" not in attrs:
        min_offset = np.floor(np.min(xyz, axis=0))
        attrs["offset"] = min_offset
    if "scales" not in attrs:
        attrs["scales"] = [0.001, 0.001, 0.001]  # 0.001 是毫米精度

    # 初始化一些需要保存的属性值。如果是固有属性，直接赋值; 如果是额外属性，添加到 header 中, 后续赋值
    extra_attr = []
    for attr, value in attrs.items():
        if hasattr(header, attr):  # 设置固有的属性的值, 如 scales, offsets
            header.__setattr__(attr, value)
        else:  # 添加额外属性，在 las 初始化后赋值
            header.add_extra_dim(laspy.ExtraBytesParams(name=attr, type=np.float32))
            extra_attr.append(attr)

    # 2. 创建 las 文件
    las = laspy.LasData(header)

    # 添加xyz坐标
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]

    # 添加RGB颜色，如果是归一化的颜色，则需要乘以 65535，转为 uint16
    # 如果RGB全是0, 则不存储颜色
    if (rgb is not None) and (np.max(rgb) > 0):
        if np.max(rgb) <= 1:
            rgb = (rgb * 65535).astype(np.uint16)  # 65535 = 2^16 - 1, las存储颜色是16位无符号整型
        elif np.max(rgb) <= 255:
            rgb = (rgb / 255 * 65535).astype(np.uint16)

        las.red = rgb[:, 0]
        las.green = rgb[:, 1]
        las.blue = rgb[:, 2]

    # 添加额外属性
    for attr in extra_attr:
        # 当 value 是 n * 1 的 ndarray 时，转换为 1 维数组
        value = attrs[attr]
        if value.ndim == 2 and value.shape[1] == 1:
            value = value.flatten()
        las[attr] = value

    # 保存LAS文件
    las.write(out_file)


def get_las_header_attrs(point_format=7, version="1.4"):
    """
    根据 point_format 和 version 获取 las 文件的 header 属性
    说明文档：https://laspy.readthedocs.io/en/latest/intro.html#point-records
    Args:
        point_format: 点格式
        version: 版本

    Returns:

    """
    dimensions = []
    header = laspy.LasHeader(point_format=point_format, version=version)  # 7 支持rgb
    for dim in header.point_format.dimensions:
        dimensions.append(dim.name)
    return dimensions


if __name__ == '__main__':
    # 测试1： 获取 las 文件头属性
    fields = get_las_header_attrs(7, "1.4")
    print(f"point_format=7, version=1.4, 头文件包含字段： {fields}")

    # 测试2： 读取LAS文件
    read_las_path = "/home/gisleung/dataset2/Epoch_March2018/LiDAR/Mar18_test_small.las"
    xyz_, rgb_, attrs_ = read_las_fit(read_las_path, ["scales", "offsets", "leung"])
    print(attrs_)

    # 测试3： 写入LAS文件
    # save_las_path = "/home/gisleung/dataset2/Epoch_March2018/LiDAR/Mar18_test_small2_fit.las"
    # write_las_fit(save_las_path, xyz_, rgb_, {
    #     # "scales": attrs_["scales"],
    #     # "offsets": attrs_["offsets"]
    # })
