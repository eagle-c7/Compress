import numpy as np
import cv2 as cv
import huffman_code as huff
import os
# DCT变换——8*8

def DCT(data):
    res = np.zeros(shape=(8, 8))
    for u in range(0,8):
        for v in range(0,8):
            FG = 0.
            temp = [0.,0.,0.,0.,0.,0.,0.,0.]
            for i in range(0,8):
                for j in range(0,8):
                    temp[i] = round(temp[i] + (float)(np.cos((2*j+1)*v*np.pi/16)*data[i][j]))
            for i in range(0,8):
                FG = round(FG + (float)(np.cos((2*i+1)*u*np.pi/16)*temp[i]))
            if u == 0:
                FG = round(FG * pow(2, 0.5) / 4)
            else:
                FG = round(FG / 2)
            if v == 0:
                FG = round(FG * pow(2, 0.5) / 4)
            else:
                FG = round(FG / 2)
            res[u][v] = FG
    return res.astype(dtype=int)

# 量化表
# Luminance_Quantization_Table
LQT = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]])

# The Chrominance Quantization Table
CQT = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]])

# 量化处理
def Quantization(data, table):
    res = np.zeros(shape=(8, 8))
    for i in range(0,8):
        for j in range(0,8):
            res[i][j] = round(data[i][j]/table[i][j])
    return res.astype(dtype=int)
    # return (round(data/table)).astype(dtype=int)

# zigzag处理
def zigzag(block):
    row = block.shape[0]
    col = block.shape[1]
    num = row * col
    list = np.zeros(num,)
    k = 0
    i = 0
    j = 0
    while i < row and j < col and k < num:
        list[k] = (int)(block.item(i,j))
        k = k + 1
        if (i + j) % 2 == 0:
            if (i-1) not in range(row) and (j+1) in range(col):
                j = j + 1
            elif (j+1) not in range(col):
                i = i + 1
            else:
                i = i - 1
                j = j + 1
        elif (i + j) % 2 == 1:
            if (i+1) in range(row) and (j-1) not in range(col):
                i = i + 1
            elif (i+1) not in range(row):
                j = j + 1
            else:
                i = i + 1
                j = j - 1
    list = list.astype(dtype=int).tolist()
    return list

# DCT变换——整张图
# 输入整张图片和对应的量化表
# 1. 分块
# 2. DCT
# 3. 量化
# 4. zigzag
# 5. 存成n * 64的矩阵并返回
# 第一个是对应层，第二个参数是量化表

def DCT_block(img, table):
    height, width = img.shape[:2]
    block_y = height // 8
    block_x = width // 8
    img_dct = np.zeros(shape=(block_x*block_y, 64))
    for h in range(block_y):
        for w in range(block_x):
          img_block = img[8*h:8*(h+1), 8*w: 8*(w+1)]
          img_dct[h*block_x+w:h*block_x+w+1, 0: ] = zigzag(Quantization(DCT(img_block), table))
    return img_dct.astype(dtype=int)

# 游程编码
def RLE(zig_res):
    res = list()
    height = zig_res.shape[0]
    for h in range(height):
        tmp = zig_res[h]
        num = 0
        for i in range(1, 64):
            if(tmp[i] != 0):
                res.append(num)
                res.append(tmp[i])
                num = 0
            else:
                num = num + 1
        res.append(0)
        res.append(0)
    return res

def to_bytes(data):    #data为字符串
    b = bytearray()      #使用bytearray存储转换结果
    end_length = len(data) % 8      #获取最后不足8位的长度
    for i in range(0, len(data) - end_length, 8):
        b.append(int(data[i:i + 8], 2))    #通过附加参数‘2’使用int函数处理8位字符串
    if end_length != 0:
        b.append(int(data[len(data) - end_length:len(data)], 2))     #写入最后不足8位的二进制
    else:
        b.append(int('0', 2))   
    b.append(int(bin(end_length), 2))   #写入最后字节有效二进制的长度
    return bytes(b)    #返回bytearray


if __name__ == "__main__":
    # read picture
    # 写报告的时候加一个文件路径输入
    # choice = input("Input 1 or 2:")
    # src_test = ""
    # res_dir = ""
    # if choice == 1:
    #     src_test = "train/test1.jpg"
    #     res_dir = "res_dir1"
    # elif choice == 2:
    #     src_test = "train/test2.jpg"
    #     res_dir = "res_dir2"

    src = cv.imread("train/test2.jpg")
    choice = input("Please choose the picture use the ID before: [0] test0; [1] test1; [2] test2; [3] your picture name:\n")

    if choice == '0':
        path = "train/test0.jpg"
        res_dir = "res_dir0"
    elif choice == '1':
        path = "train/test1.jpg"
        res_dir = "res_dir1"
    elif choice == '2':
        path = "train/test2.jpg"
        res_dir = "res_dir2"
    else:
        path = input("Please input your picture path:\n")
        res_dir = input("Please input your res direction:\n")

    print("Compressing...")
    src = cv.imread(path)
    # RGB->YUV
    # 按照yuv读取图片
    yuv = cv.cvtColor(src, cv.COLOR_BGR2YUV)
    #存储：高 宽 YUV
    img = np.array(yuv)
    img_h = img.shape[0]
    img_w = img.shape[1]

    # YUV分别存储
    img_Y = np.zeros((img_w, img_h))
    img_U = np.zeros((img_w, img_h))
    img_V = np.zeros((img_w, img_h))
    for i in range(0, img_w):
        for j in range(0, img_h):
            temp = img[j,i]
            img_Y[i,j] = temp[0]
            img_U[i,j] = temp[1]
            img_V[i,j] = temp[2]

    # 由于后期下采样和DCT处理的需要，这里提前把图片的宽高扩展成16的倍数
    # 这里扩充的方式是直接复制最后一行和最后一列
    while img_Y.shape[0]%16 != 0:
        img_Y = np.insert(img_Y, img_Y.shape[0], values=img_Y[img_Y.shape[0]-1], axis=0)
        img_U = np.insert(img_U, img_U.shape[0], values=img_U[img_U.shape[0]-1], axis=0)
        img_V = np.insert(img_V, img_V.shape[0], values=img_V[img_V.shape[0]-1], axis=0)
    while img_Y.shape[1]%16 != 0:
        img_Y = np.insert(img_Y, img_Y.shape[1], values=img_Y.T[img_Y.shape[1]-1], axis=1)
        img_U = np.insert(img_U, img_U.shape[1], values=img_U.T[img_U.shape[1]-1], axis=1)
        img_V = np.insert(img_V, img_V.shape[1], values=img_V.T[img_V.shape[1]-1], axis=1)
    
    # subsample
    # 按照YUV420的格式，每2*2的Y对应一个U和V
    img_U = np.delete(np.delete(img_U, np.s_[1::2], 1), np.s_[1::2], 0)
    img_V = np.delete(np.delete(img_V, np.s_[1::2], 1), np.s_[1::2], 0)
    # 对AC进行游程编码
    # 把DC全部拉出来，进行DPCM
    # 展开成一维信号序列
    zig_Y = DCT_block(img_Y, LQT)
    zig_U = DCT_block(img_U, CQT)
    zig_V = DCT_block(img_V, CQT)
    # print(zig_Y)
    # RLE on AC
    # 得到的结果是列表的形式
    AC_Y = RLE(zig_Y)
    AC_U = RLE(zig_U)
    AC_V = RLE(zig_V)
    # DPCM on DC coefficients
    # 得到的结果是列表的形式
    DC_Y = (zig_Y[:, 0]).tolist()
    DC_U = (zig_U[:, 0]).tolist()
    DC_V = (zig_V[:, 0]).tolist()
    for i in range(len(DC_Y)-1, 0, -1):
        DC_Y[i] = DC_Y[i-1] - DC_Y[i]
    for i in range(len(DC_U)-1, 0, -1):
        DC_U[i] = DC_U[i-1] - DC_U[i]
        DC_V[i] = DC_V[i-1] - DC_V[i]
    # perform entropy coding
    # 返回字典——数：编码
    # 每层分别编码
    huff_AC_Y = huff.build_table(AC_Y)
    huff_AC_U = huff.build_table(AC_U)
    huff_AC_V = huff.build_table(AC_V)
    huff_DC_Y = huff.build_table(DC_Y)
    huff_DC_U = huff.build_table(DC_U)
    huff_DC_V = huff.build_table(DC_V)

    # 拉成01串处理
    AC_Y_list = ""
    AC_U_list = ""
    AC_V_list = ""

    for i in AC_Y:
        AC_Y_list = AC_Y_list + huff_AC_Y.get(i)
    for i in AC_U:
        AC_U_list = AC_U_list + huff_AC_U.get(i)
    for i in AC_V:
        AC_V_list = AC_V_list + huff_AC_V.get(i)

    DC_Y_list = ""
    DC_U_list = ""
    DC_V_list = ""

    for i in DC_Y:
        DC_Y_list = DC_Y_list + huff_DC_Y.get(i)
    for i in DC_U:
        DC_U_list = DC_U_list + huff_DC_U.get(i)
    for i in DC_V:
        DC_V_list = DC_V_list + huff_DC_V.get(i)


    os.makedirs(res_dir)

    d_info = {}
    d_info.update({"h":img.shape[0]})
    d_info.update({"w":img.shape[1]})
    np.save(res_dir+"/info.npy",d_info)
    # 
    bin_file = open(res_dir+"/s_AC_Y", mode='wb')
    bin_file.write(to_bytes(AC_Y_list))
    bin_file = open(res_dir+"/s_AC_U", mode='wb')
    bin_file.write(to_bytes(AC_U_list))
    bin_file = open(res_dir+"/s_AC_V", mode='wb')
    bin_file.write(to_bytes(AC_V_list))

    bin_file = open(res_dir+"/s_DC_Y", mode='wb')
    bin_file.write(to_bytes(DC_Y_list))
    bin_file = open(res_dir+"/s_DC_U", mode='wb')
    bin_file.write(to_bytes(DC_U_list))
    bin_file = open(res_dir+"/s_DC_V", mode='wb')
    bin_file.write(to_bytes(DC_V_list))
    
    np.save(res_dir+"/huff_AC_Y.npy",huff_AC_Y)
    np.save(res_dir+"/huff_AC_U.npy",huff_AC_U)
    np.save(res_dir+"/huff_AC_V.npy",huff_AC_V)
    np.save(res_dir+"/huff_DC_Y.npy",huff_DC_Y)
    np.save(res_dir+"/huff_DC_U.npy",huff_DC_U)
    np.save(res_dir+"/huff_DC_V.npy",huff_DC_V)
    print("Compress Over")
    