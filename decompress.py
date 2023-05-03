from PIL import Image
import numpy as np
import cv2 as cv 
import huffman_code as huff
import math

def read_str(src):
    bin_file = open(src, mode='rb')
    bin_str=bin_file.read()
    word_bin = format(int.from_bytes(bin_str, byteorder='big', signed=False),'#0' + str(len(bin_str) * 8 + 2) + 'b')[2:]
    end_length_bin = word_bin[len(word_bin) - 8:len(word_bin)]
    end_length = int(end_length_bin, 2)
    word_bin = word_bin[0:-8]
    word_bin = word_bin[0:-8] + word_bin[len(word_bin) -end_length:len(word_bin)]
    return word_bin

def i_RLE(n, l, bn):
    iz = np.zeros(shape=(bn, 63))
    temp = list()
    for i in range(0,len(l),2):
        if l[i+1] == 0:
            for j in range(0, 63-len(temp)):
                temp.append(0)
            iz[n] = np.array(temp)
            n = n + 1
            temp = list()
            continue
        for j in range(0, (int)(l[i])):
            temp.append(0)
        temp.append(l[i+1])
    return iz

def i_zigzag(list):
    row = 8
    col = 8
    num = row * col
    block = np.zeros(shape=(row,col))
    k = 0
    i = 0
    j = 0
    while i < row and j < col and k < num:
        block[i][j] = list[k]
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
    return block

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

# 反量化
def i_quantization(data, table):
    res = np.zeros(shape=(8, 8))
    for i in range(0,8):
        for j in range(0,8):
            res[i][j] = data[i][j] * table[i][j]
    return res.astype(dtype=int)

# 处理8*8
def i_DCT(block):
    res = np.zeros(shape=(8,8))
    for i in range(0,8):
        for j in range(0,8):
            FG = 0.
            temp = [0.,0.,0.,0.,0.,0.,0.,0.]
            for u in range(0,8):
                for v in range(0,8):
                    if v == 0:
                        c = pow(2, 0.5) / 4
                    else:
                        c = 0.5
                    temp[u] = temp[u]+c*block[u][v]*np.cos((2*j+1)*v*np.pi/16)
                if u == 0:
                    c = pow(2, 0.5) / 4
                else:
                    c = 0.5
                temp[u] = temp[u]*np.cos((2*i+1)*u*np.pi/16)*c
                FG = FG + temp[u]
            res[i][j] = round(FG)
    return res.astype(dtype=int)


# 处理整张图片
def i_DCT_block(height, width, img_dct, table):
    block_y = height // 8
    block_x = width // 8
    img = np.zeros(shape=(width, height))
    for i in range(img_dct.shape[0]):
        img_block = img_dct[i:i+1, 0: ]
        w = i // block_y
        h = i % block_y
        img[8*w: 8*(w+1), 8*h:8*(h+1)] = i_DCT(i_quantization(i_zigzag(img_block[0].astype(dtype=int).tolist()), table))
    # img = np.zeros(shape=(height, width))
    # for h in range(block_y):
    #     for w in range(block_x):
    #         img_block = img_dct[h*block_x+w:h*block_x+w+1, 0: ]
    #         img[8*h:8*(h+1), 8*w: 8*(w+1)] = i_DCT(i_quantization(i_zigzag(img_block[0].astype(dtype=int).tolist()), table))
    # print(img.shape[0])
    return img.astype(dtype=int)

def yuv2rgb(img_Y,img_U,img_V, height, width):
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    #恢复原大小
    img_U = np.repeat(np.repeat(img_U, 2, 0), 2, 1)
    img_V = np.repeat(np.repeat(img_V, 2, 0), 2, 1)

    c = (img_Y-np.array([16])) * 298
    d = img_U - np.array([128])
    e = img_V - np.array([128])

    r = (c + 409 * e + 128) // 256
    g = (c - 100 * d - 208 * e + 128) // 256
    b = (c + 516 * d + 128) // 256

    r = np.where(r < 0, 0, r)
    r = np.where(r > 255,255,r)

    g = np.where(g < 0, 0, g)
    g = np.where(g > 255,255,g)

    b = np.where(b < 0, 0, b)
    b = np.where(b > 255,255,b)

    rgb[:, :, 2] = r
    rgb[:, :, 1] = g
    rgb[:, :, 0] = b

    return rgb


if __name__ == "__main__":

    choice = input("Please choose the picture use the ID before: [0] test0; [1] test1; [2] test2; [3] your picture name:\n")

    if choice == '0':
        save_src = "train/test_res0.jpg"
        load_src = "res_dir0"
    elif choice == '1':
        save_src = "train/test_res1.jpg"
        load_src = "res_dir1"
    elif choice == '2':
        save_src = "train/test_res2.jpg"
        load_src = "res_dir2"
    else:
        load_src = input("Please input your picture path:\n")
        save_src = input("Please input your res direction:\n")
    

    print("Decompressing...")
    info = np.load(load_src + "/info.npy",allow_pickle = True).item()

    huff_AC_Y = np.load(load_src + "/huff_AC_Y.npy",allow_pickle = True).item()
    huff_AC_U = np.load(load_src + "/huff_AC_U.npy",allow_pickle = True).item()
    huff_AC_V = np.load(load_src + "/huff_AC_V.npy",allow_pickle = True).item()
    huff_DC_Y = np.load(load_src + "/huff_DC_Y.npy",allow_pickle = True).item()
    huff_DC_U = np.load(load_src + "/huff_DC_U.npy",allow_pickle = True).item()
    huff_DC_V = np.load(load_src + "/huff_DC_V.npy",allow_pickle = True).item()

    # print(huff_AC_V)
    s_AC_Y = read_str(load_src + "/s_AC_Y")
    s_AC_U = read_str(load_src + "/s_AC_U")
    s_AC_V = read_str(load_src + "/s_AC_V")
    s_DC_Y = read_str(load_src + "/s_DC_Y")
    s_DC_U = read_str(load_src + "/s_DC_U")
    s_DC_V = read_str(load_src + "/s_DC_V")
    # # print(s_AC_V)
    # print(len(s_DC_Y))
    AC_Y = huff.decode(s_AC_Y,huff_AC_Y)
    AC_U = huff.decode(s_AC_U,huff_AC_U)
    AC_V = huff.decode(s_AC_V,huff_AC_V)
    DC_Y = huff.decode(s_DC_Y,huff_DC_Y)
    DC_U = huff.decode(s_DC_U,huff_DC_U)
    DC_V = huff.decode(s_DC_V,huff_DC_V)
    # print(DC_Y)
    # print(len(DC_Y))

    height = math.ceil(info.get("h") / 16) * 16
    width = math.ceil(info.get("w") / 16) * 16


    # DC
    for i in range(1, len(DC_Y)):
        DC_Y[i] = DC_Y[i-1] - DC_Y[i]
    for i in range(1, len(DC_U)):
        DC_U[i] = DC_U[i-1] - DC_U[i]
        DC_V[i] = DC_V[i-1] - DC_V[i]

    # AC 反游程编码
    block_num_Y = len(DC_Y)
    block_num_UV = len(DC_U)

    i_zig_Y = i_RLE(0, AC_Y, block_num_Y)
    i_zig_U = i_RLE(0, AC_U, block_num_UV)
    i_zig_V = i_RLE(0, AC_V, block_num_UV)

    # print(i_zig_Y.shape[:])
    # print(len(DC_Y))
    i_zig_Y = np.insert(i_zig_Y, 0, DC_Y, axis=1)
    i_zig_U = np.insert(i_zig_U, 0, DC_U, axis=1)
    i_zig_V = np.insert(i_zig_V, 0, DC_V, axis=1)
    
    # 反zigzag
    # 反DCT
    img_Y = i_DCT_block(height, width, i_zig_Y, LQT)
    img_U = i_DCT_block(height//2, width//2, i_zig_U, CQT)
    img_V = i_DCT_block(height//2, width//2, i_zig_V, CQT)


    res = yuv2rgb(img_Y,img_U,img_V, width, height)
    # 三层合并，返回RGB
    # test output
    im = Image.new("RGB", (info.get("w"), info.get("h")))
    for i in range(0, info.get("w")):
        for j in range(0, info.get("h")):
            im.putpixel((i,j), ((int)(res[i][j][2]),(int)(res[i][j][1]),(int)(res[i][j][0])))
    im.save(save_src)
    print("Decompress Over")