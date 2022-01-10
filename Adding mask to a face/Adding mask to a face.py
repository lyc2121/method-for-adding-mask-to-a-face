import cv2
import numpy as np
import random
from tqdm import tqdm
import os
import math

img_path_in = 'I:/106facedata/PFLD-106-master/data/train_data/list.txt'
img_path_out = "I:/106facedata/PFLD-106-master/data/train_data/img_mask"
file_path_out = "I:/106facedata/PFLD-106-master/data/train_data/list_mask69300-70000.txt"

img_path_in_1 = 'I:/106facedata/PFLD-106-master/data/test_data/list.txt'
img_path_out_1 = "I:/106facedata/PFLD-106-master/data/test_data/img_mask"
file_path_out_1 = "I:/106facedata/PFLD-106-master/data/test_data/list_mask0-2640.txt"

if not os.path.exists(img_path_out):
    os.makedirs(img_path_out)

key_point_poly_down=[11,12,13,14,15,16,2,3,4,5,6,7,8,0,24,23,22,21,20,19,18,32,31,30,29,28,27,74]
# key_point_poly_down=[11,15,16,2,3,4,5,6,7,8,0,24,23,22,21,20,19,18,32,74]

key_point_face_simple=[1,16,0,32,17,104,49]

wgs=[13,14,15,16,2,3,4,5,6,7,8,0,24,23,22,21,20,19,18,32,31,30,29,76,74,86,82,77,78,79,80,85,84,83,52,71,53,61,62,60,52,65,64,55,66,54,63,56,71,62,60,53,67,59,70,57,68,58,69,61]

key_point_poly_74_10_11=[10,11,74]
key_point_poly_74_26_27=[26,27,74]

key_point_poly_up=[]

def rayCasting(p, poly):
    px = p[0]
    py = p[1]
    flag = False

    i = 0
    l = len(poly)
    j = l - 1
    # for(i = 0, l = poly.length, j = l - 1; i < l; j = i, i++):
    while i < l:
        sx = poly[i]['x']
        sy = poly[i]['y']
        tx = poly[j]['x']
        ty = poly[j]['y']

        # 点与多边形顶点重合
        if (sx == px and sy == py) or (tx == px and ty == py):
            return (px, py)

        # 判断线段两端点是否在射线两侧
        if (sy < py and ty >= py) or (sy >= py and ty < py):
            # 线段上与射线 Y 坐标相同的点的 X 坐标
            x = sx + (py - sy) * (tx - sx) / (ty - sy)
            # 点在多边形的边上
            if x == px:
                return (px, py)
            # 射线穿过多边形的边界
            if x > px:
                flag = not flag
        j = i
        i += 1

    # 射线穿过多边形边界的次数为奇数时点在多边形内
    return (px, py) if flag else 'out'


# 根据数组下标奇偶数得到点的坐标
def getpoint(a):
    i = 0
    zhima = []
    while i < len(a.split(',')[1::2]):
        zhima.append({'x': float(a.split(',')[::2][i]), 'y': float(a.split(',')[1::2][i])})
        i += 1
    return zhima


# 根据输入的点循环判断芝麻是否在多边形里面，如果全部在外面则输出no,否则输出芝麻的坐标
def rs(point, duobianxing):
    dbx = getpoint(duobianxing)
    rs = rayCasting(point, dbx)
    if rs == 'out':
        return 0
    else:
        return 1


def get_avger_light(img,ploy_face):
    color=0
    rows, cols, channel = img.shape
    m=0
    for i in range(rows):

        for j in range(cols):

            for c in range(3):
                point=[j,i]
                is_in = rs(point, ploy_face)
                if is_in:
                    color = img[i, j][c] + color
                    m=m+1
    avrage_light=color/m
    return round(avrage_light,0)





def change_light(img,a,b,WG,img_light_add):
    rows,cols,channel=img.shape
    dst = img.copy()
    for i in range(rows):

        for j in range(cols):

            for c in range(3):
                dis=math.sqrt((i-WG[1])**2+(j-WG[0])**2)
                if dis==0:
                    rate=1
                else:
                    rate = 40 / dis
                color = int((img[i, j][c] +img_light_add)* a*rate + b)

                if color > 255:

                    color = 255

                elif color < 0:

                    color = 0
                dst[i, j][c]=color
    return dst
def  change_light_intelligent(img,a,b,WG,duobianxing):
    rows, cols, channel = img.shape
    dst = img.copy()
    for i in range(rows):

        for j in range(cols):
            for c in range(3):

                point=[j,i]

                is_in=rs(point, duobianxing)

                if is_in:
                    color = 255
                else:
                    color=img[i, j][c]
                if color > 255:

                    color = 255

                elif color < 0:

                    color = 0
                dst[i, j][c] = color
    return dst


def BGR_TO_GRAY(path):
    img = cv2.imread(path)
    Grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_final = cv2.cvtColor(Grayimg, cv2.COLOR_GRAY2RGB)
    return img_final
def BGR_TO_RGB(path):
    img = cv2.imread(path)
    RGBimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return RGBimg

def get_ploy(landmaks,key_point_poly):
    xy_poly = []
    for i in key_point_poly:
        x_index = 2 * i
        y_index = 2 * i + 1
        xy_poly.append(landmaks[x_index] * w)
        xy_poly.append(landmaks[y_index] * h)
    poly_input = ''
    isfirst = True
    for i in xy_poly:
        if isfirst:
            poly_input = str(i)
            isfirst = False
        else:
            poly_input = poly_input + "," + str(i)
    return poly_input
data_all=[]
epcoh=0
with open(img_path_in) as f:
    datas = f.readlines()
    # randomByteArray1 = bytearray(os.urandom(37632))
    # flatNumpyArray1 = np.array(randomByteArray1)
    # BGRimage = flatNumpyArray1.reshape(112, 112, 3)
    random.shuffle(datas)
    item=[]
    for data_line in tqdm(datas[69300:70000]):
        epcoh=epcoh+1
        item = [i for i in data_line.strip('\n').strip().split(' ')]
        img_path_in_item = item[0]
        name=str(img_path_in_item).split('/')[-1]
        landmaks=[float(i) for i in item[1:]]



        img=BGR_TO_GRAY(img_path_in_item)
        h,w,_=img.shape
        ploy_down_simple=get_ploy(landmaks,key_point_face_simple)
        img_light=get_avger_light(img,ploy_down_simple)
        img_light_add=random.randint(120,170)-img_light


        wgs_index=random.randint(0,60)
        wg =[landmaks[wgs_index*2]*w,landmaks[wgs_index*2+1]*h]

        img_out_1=change_light(img,1.5,10,wg,img_light_add)
        ploy_down=get_ploy(landmaks,key_point_poly_down)
        img_out_2=change_light_intelligent(img_out_1,3,10,wg,ploy_down)
        img_out_3= cv2.GaussianBlur(img_out_2, ksize=(3, 3), sigmaX=0, sigmaY=0)

        path_out=img_path_out+'/'+name
        cv2.imwrite(path_out,img_out_3)
        item.append(path_out)

        for line in landmaks:
            item.append(line)
        data_all.append(item)

        # if epcoh%100==0:
        #     with open(file_path_out, 'w') as f_out:
        #         for line in data_all:
        #             for line_item in line:
        #                 f_out.write(str(line_item) + ' ')
        #             f_out.write('\n')
        #         f_out.close()




        # cv2.namedWindow('0',2)
        # cv2.imshow('0',img_out_3)
        # cv2.waitKey(0)
