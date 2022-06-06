import sys
import numpy as np
from imageio import imread, imwrite
from matplotlib import pyplot as plt
from scipy.ndimage.filters import convolve
# 图像处理时候查看处理进度条
from tqdm import trange


# 计算能量图
def calc_energy(img):
    # 这里是X和Y轴的sobel算子
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0]])
    # RGB三颜色通道，为每个通道都复制一份相同的滤波器
    filter_du = np.stack([filter_du] * 3, axis=2)
    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0]])
    filter_dv = np.stack([filter_dv] * 3, axis=2)
    print(type(img))
    img = img.astype('float32')
    # 然后根据论文中的能量最基本的公式计算原始图像的能量图
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))
    # 将RGB三颜色通道的能量进行相加得到原始图的能量图
    energy_map = convolved.sum(axis=2)
    return energy_map


# 求最小接缝
def minimum_seam(img):
    row, column, _ = img.shape
    energy_map = calc_energy(img)
    arr = energy_map.copy()
    # backtrack = np.zeros_like(arr, dtype=np.int)
    # 从第二行开始
    for i in range(1, row):
        for j in range(0, column):
            # 这里处理左侧边缘部分，确保数组不会越界
            # 使用动态规划算法求解最小值
            if j == 0:
                # index = np.argmin(arr[i - 1, j:j + 2])
                # backtrack[i, j] = index + j
                # min_energy = arr[i - 1, index + j]
                min_energy = min(arr[i][j], arr[i][j+1])
            elif j == column - 1:
                # index = np.argmin(arr[i - 1, j-1:j + 1])
                # backtrack[i, j] = index + j - 1
                # min_energy = arr[i - 1, index + j - 1]
                min_energy = min(arr[i][j-1], arr[i][j-1])
            else:
                # index = np.argmin(arr[i - 1, j - 1:j + 2])
                # backtrack[i, j] = index + j - 1
                # min_energy = arr[i - 1, index + j - 1]
                min_energy = min(arr[i][j - 1], arr[i][j], arr[i][j + 1])

            arr[i, j] += min_energy

    return arr


# @numba.jit
def carve_column(img):
    row, column, _ = img.shape
    arr = minimum_seam(img)
    # 创建一个和原始图一样大小的矩阵，初始化为True
    mask = np.ones((row, column), dtype=np.bool)
    # 逐行找到最小元素的位置，采用的是动态规划算法
    j = np.argmin(arr[-1])
    for i in reversed(range(row)):
        # 把需要删除的像素位置标记
        mask[i, j] = False
        if i > 0:
            if j == 0:
                j += np.argmin(arr[i - 1, j:j+2])
            elif j == column-1:
                j += np.argmin(arr[i - 1, j-1:j+1]) - 1
            else:
                j += np.argmin(arr[i - 1, j-1:j+2]) - 1
    # RGB三颜色通道同时标记
    mask = np.stack([mask] * 3, axis=2)
    # reshape成和原始图像大小一致的维度
    img = img[mask].reshape((row, column - 1, 3))
    return img, mask, arr


def crop_c(img, scale_c):
    row, column, _ = img.shape
    new_column = int(scale_c * column)
    # mask, M = None
    # 删除每次的最小能量线， 由scale_c控制做几次
    for i in trange(column - new_column):
        img, _, _ = carve_column(img)
    return img


def crop_r(img, scale_r):
    # 对原始图像矩阵进行旋转90°
    img = np.rot90(img, 1, (0, 1))
    # 本来进行列的删除，现在旋转后进行行的删除，然后在做矩阵的三次旋转旋转回来
    img = crop_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img


def main():
    # 从控制台获取输入
    if len(sys.argv) != 5:
        print('<文件名> <坐标轴> <scale 范围：0~1> <image_in> <image_out>', file=sys.stderr)
        sys.exit(1)
    axis = sys.argv[1]
    scale = float(sys.argv[2])
    in_filename = sys.argv[3]
    out_filename = sys.argv[4]
    img = imread(in_filename)
    r, c, _ = img.shape
    if axis == 'r':
        out = crop_r(img, scale)
    elif axis == 'c':
        out = crop_c(img, scale)
    else:
        print('<坐标轴> 请输入 “r" 或者 "c"', file=sys.stderr)
        sys.exit(1)

    imwrite(out_filename, out)


if __name__ == '__main__':
    main()
    img = imread('test2.jpg')
    # 展示能量图
    energy = calc_energy(img)
    plt.imshow(energy)
    plt.show()
