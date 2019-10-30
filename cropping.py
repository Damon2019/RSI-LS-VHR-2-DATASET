#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Saturday June 29 10:09:02 2019
@author: Damon

"""
import cv2
import random
import os
import os.path
import pandas as pd
import numpy  as np
from copy import deepcopy
from lxml.etree import Element, SubElement, tostring
import pprint
from xml.dom.minidom import parseString
from skimage import io,transform

# 每两个检测框框是否有交叉，如果有交集则返回重叠度 IOU, 如果没有交集则返回 0
def area_overlab(x1, y1, w1, h1, data):
    '''
    :第一幅框为滑动窗口的大小
    :param x1: 滑动窗口的左上角 x 坐标
    :param y1: 滑动窗口的左上角 y 坐标
    :param w1: 滑动窗口中的检测框的宽度
    :param h1: 滑动窗口中的检测框的高度
    :return: 两个如果有交集则返回重叠度 IOU, 如果没有交集则返回 0
    '''
    overlapRate = [0]
    for j in range(len(data)):
        x2 = data.values[j][0]
        y2 = data.values[j][1]
        w2 = data.values[j][2]
        h2 = data.values[j][3]

        if(x1>x2+w2):
            continue
        if(y1>y2+h2):
            continue
        if(x1+w1<x2):
            continue
        if(y1+h1<y2):
            continue
        colInt = abs(min(x1 +w1 ,x2+w2) - max(x1, x2))
        rowInt = abs(min(y1 + h1, y2 +h2) - max(y1, y2))
        overlap_area = colInt * rowInt
        area1 = w1 * h1
        area2 = w2 * h2
        #overlapRate 计算得出的重叠率
        if(overlap_area / area2 > 0):
            overlapRate.append(overlap_area / area2)

    return overlapRate


# 和data.values[j][4]比较，判断是不是0 和 1
def area_overlab_compare_dat(x1, y1, w1, h1, data):
    flag_4 = 0
    overlapRate = [0]
    for j in range(len(data)):
        x2 = data.values[j][0]
        y2 = data.values[j][1]
        w2 = data.values[j][2]
        h2 = data.values[j][3]
        gather = data.values[j][4]

        if(x1>x2+w2):
            continue
        if(y1>y2+h2):
            continue
        if(x1+w1<x2):
            continue
        if(y1+h1<y2):
            continue
        colInt = abs(min(x1 +w1 ,x2+w2) - max(x1, x2))
        rowInt = abs(min(y1 + h1, y2 +h2) - max(y1, y2))
        overlap_area = colInt * rowInt
        area1 = w1 * h1
        area2 = w2 * h2
        #overlapRate 计算得出的重叠率
        if(overlap_area / area2 > 0):
            overlapRate.append(overlap_area / area2)
            # 一旦有聚集则使flag_4 = false
            if gather == 1:
                flag_4 = 1

    return overlapRate, flag_4

# 每两个检测框框是否有交叉，如果有交集则返回重叠度 IOU, 如果没有交集则返回 0
# 计算位置
def area_overlab_double(x1, y1, w1, h1, x2, y2, w2, h2):

    if(x1>x2+w2):
        return 0
    if(y1>y2+h2):
        return 0
    if(x1+w1<x2):
        return 0
    if(y1+h1<y2):
        return 0
    colInt = abs(min(x1 +w1 ,x2+w2) - max(x1, x2))
    rowInt = abs(min(y1 + h1, y2 +h2) - max(y1, y2))
    overlap_area = colInt * rowInt
    area1 = w1 * h1
    area2 = w2 * h2
    overlapRate = overlap_area / area2

    return overlapRate

#-------------------------------------------------------------------------------------------
#扩大范围

#左边的线
def xl_location(slide_x1, slide_y1, x_bSteps, y_bSteps, data, expand_i, bStep):
    compareOverlapRate = [0]
    # 判断直线和矩形框是否有相交
    listD = [0]
    collectD = [-1]
    fStep = 0
    # bStep = 10
    flag = 0 #表示全部都满足条件的状态

    #扩大改进算法
    # 每组每条边和矩形框交到的个数
    iCount = 0
    iCountD = [100]
    overlapRate_double = [0]
    null = [0]
    #overlapRate_double = [null]
    overlapRate_All = [null]
    finally_All = [null]
    listCollectD = []

    if expand_i==1 or expand_i==2 or expand_i==3 or expand_i==4:
        allStep = 201
    elif expand_i==5 or expand_i==6 or expand_i==7 or expand_i==8:
        allStep = 101


    for step in range(0,allStep,bStep):
        for k in range(len(data)):
            fStep = step
            #扩大
            if expand_i == 1:
                l1 = [slide_x1 - step, slide_y1]
                l2 = [slide_x1 - step, slide_y1 + y_bSteps]
                sq = [data.values[k][0], data.values[k][1]+data.values[k][3], data.values[k][0]+data.values[k][2], data.values[k][1]]
            elif expand_i == 2:
                l1 = [slide_x1, slide_y1 - step]
                l2 = [slide_x1 + x_bSteps, slide_y1 - step]
                sq = [data.values[k][0], data.values[k][1]+data.values[k][3], data.values[k][0]+data.values[k][2], data.values[k][1]]
            elif expand_i == 3:
                l1 = [slide_x1 + x_bSteps + step, slide_y1]
                l2 = [slide_x1 + x_bSteps + step, slide_y1 + y_bSteps]
                sq = [data.values[k][0], data.values[k][1]+data.values[k][3], data.values[k][0]+data.values[k][2], data.values[k][1]]
            elif expand_i == 4:
                l1 = [slide_x1, slide_y1 + y_bSteps + step]
                l2 = [slide_x1 + x_bSteps, slide_y1 + y_bSteps + step]
                sq = [data.values[k][0], data.values[k][1]+data.values[k][3], data.values[k][0]+data.values[k][2], data.values[k][1]]
            elif expand_i == 5:
                l1 = [slide_x1 + step, slide_y1]
                l2 = [slide_x1 + step, slide_y1 + y_bSteps]
                sq = [data.values[k][0], data.values[k][1]+data.values[k][3], data.values[k][0]+data.values[k][2], data.values[k][1]]
            elif expand_i == 6:
                l1 = [slide_x1, slide_y1 + step]
                l2 = [slide_x1 + x_bSteps, slide_y1 + step]
                sq = [data.values[k][0], data.values[k][1]+data.values[k][3], data.values[k][0]+data.values[k][2], data.values[k][1]]
            elif expand_i == 7:
                l1 = [slide_x1 + x_bSteps - step, slide_y1]
                l2 = [slide_x1 + x_bSteps - step, slide_y1 + y_bSteps]
                sq = [data.values[k][0], data.values[k][1]+data.values[k][3], data.values[k][0]+data.values[k][2], data.values[k][1]]
            elif expand_i == 8:
                l1 = [slide_x1, slide_y1 + y_bSteps - step]
                l2 = [slide_x1 + x_bSteps, slide_y1 + y_bSteps - step]
                sq = [data.values[k][0], data.values[k][1]+data.values[k][3], data.values[k][0]+data.values[k][2], data.values[k][1]]

            D = check(l1, l2, sq)
            #if D == 0:
            listD.append(D)
            if D == 1:
                iCount += 1

                if expand_i == 1:
                    overlapRate = area_overlab_double(slide_x1-step, slide_y1, x_bSteps+step, y_bSteps, data.values[k][0], data.values[k][1], data.values[k][2], data.values[k][3])
                    overlapRate_double.append(overlapRate)
                elif expand_i == 2:
                    overlapRate = area_overlab_double(slide_x1, slide_y1-step, x_bSteps, y_bSteps+step, data.values[k][0], data.values[k][1], data.values[k][2], data.values[k][3])
                    overlapRate_double.append(overlapRate)
                elif expand_i == 3:
                    overlapRate = area_overlab_double(slide_x1, slide_y1, x_bSteps+step, y_bSteps, data.values[k][0], data.values[k][1], data.values[k][2], data.values[k][3])
                    overlapRate_double.append(overlapRate)
                elif expand_i == 4:
                    overlapRate = area_overlab_double(slide_x1, slide_y1, x_bSteps, y_bSteps+step, data.values[k][0], data.values[k][1], data.values[k][2], data.values[k][3])
                    overlapRate_double.append(overlapRate)
                elif expand_i == 5:
                    overlapRate = area_overlab_double(slide_x1+step, slide_y1, x_bSteps-step, y_bSteps, data.values[k][0], data.values[k][1], data.values[k][2], data.values[k][3])
                    overlapRate_double.append(overlapRate)
                elif expand_i == 6:
                    overlapRate = area_overlab_double(slide_x1, slide_y1+step, x_bSteps, y_bSteps-step, data.values[k][0], data.values[k][1], data.values[k][2], data.values[k][3])
                    overlapRate_double.append(overlapRate)
                elif expand_i == 7:
                    overlapRate = area_overlab_double(slide_x1, slide_y1, x_bSteps-step, y_bSteps, data.values[k][0], data.values[k][1], data.values[k][2], data.values[k][3])
                    overlapRate_double.append(overlapRate)
                elif expand_i == 8:
                    overlapRate = area_overlab_double(slide_x1, slide_y1, x_bSteps, y_bSteps-step, data.values[k][0], data.values[k][1], data.values[k][2], data.values[k][3])
                    overlapRate_double.append(overlapRate)

        if(max(listD) == 0):
             flag = 1
             break
        else :
            # 计算个数，初始化
            iCountD.append(iCount)
            iCount = 0
            overlapRate_All.append(deepcopy(overlapRate_double))
            overlapRate_double.clear()
            overlapRate_double.append(0)

            listD.clear()
            listD.append(0)

    minIndex = expand_all(iCountD, overlapRate_All)

    minIndex -= 1

    if flag == 1 :
        #print('我执行在flag里面',fStep)
        return fStep
    else:
        #print('我执行在xl minIndex里面------------------',minIndex)
        return bStep * minIndex



#----------------------------------------------------------------------

def expand_all(iCountD, overlapRate_All):
    listCollectD = []
    # 选取iCountD中最小的，并获得全部坐标
    min_iCountD = min(iCountD)
    for i in range(len(iCountD)):
        if min_iCountD == iCountD[i]:
            #获得最小的坐标
            listCollectD.append(i)

    minRate = 2
    minIndex = 10000

    if min_iCountD == 1:    #边界线切到多个目标框
        #都变成<0.5的数字，取最小
        for j in range(len(listCollectD)):
            number = listCollectD[j]
            if overlapRate_All[number][1] > 0.5 and overlapRate_All[number][1] < 1 :
                overlapRate_All[number][1] = 1- overlapRate_All[number][1]

        for k in range(len(listCollectD)):
            number = listCollectD[k]
            if overlapRate_All[number][1] < minRate:
                minRate = overlapRate_All[number][1]
                minIndex = listCollectD[k]
    else:   #边界线切到多个目标框
        #都变成<0.5的数字，取最小
        for j in range(len(listCollectD)):
            number = listCollectD[j]
            for z in range(len(overlapRate_All[number])):
                if overlapRate_All[number][z] > 0.5 and overlapRate_All[number][z] < 1 :
                    overlapRate_All[number][z] = 1- overlapRate_All[number][z]

        for k in range(len(listCollectD)):
            number = listCollectD[k]
            sum = 0
            for z in range(len(overlapRate_All[number])):
                sum += overlapRate_All[number][z]
            overlapRate_All[number][0] = sum

        for f in range(len(listCollectD)):
            number = listCollectD[f]
            if overlapRate_All[number][0] < minRate:
                minRate = overlapRate_All[number][0]
                minIndex = listCollectD[k]
    return minIndex


#根据直线求与矩形框是否相交
def check(l1,l2,sq):    #sq 代表矩形左下角和右上角的坐标
    # step 1 check if end point is in the square
    if ( l1[0] >= sq[0] and l1[1] <= sq[1] and  l1[0] <= sq[2] and  l1[1] >= sq[3]) or ( l2[0] >= sq[0] and l2[1] <= sq[1] and  l2[0] <= sq[2] and  l2[1] >= sq[3]):
        return 1
    else:
        # step 2 check if diagonal cross the segment
        p1 = [sq[0],sq[1]]
        p2 = [sq[2],sq[3]]
        p3 = [sq[2],sq[1]]
        p4 = [sq[0],sq[3]]
        if segment(l1,l2,p1,p2) or segment(l1,l2,p3,p4):
            return 1
        else:
            return 0

def cross(p1,p2,p3): # 叉积判定
    x1=p2[0]-p1[0]
    y1=p2[1]-p1[1]
    x2=p3[0]-p1[0]
    y2=p3[1]-p1[1]
    return x1*y2-x2*y1

def segment(p1,p2,p3,p4): #判断两线段是否相交
    #矩形判定，以l1、l2为对角线的矩形必相交，否则两线段不相交
    if(max(p1[0],p2[0])>=min(p3[0],p4[0]) #矩形1最右端大于矩形2最左端
    and max(p3[0],p4[0])>=min(p1[0],p2[0]) #矩形2最右端大于矩形1最左端
    and max(p1[1],p2[1])>=min(p3[1],p4[1]) #矩形1最高端大于矩形2最低端
    and max(p3[1],p4[1])>=min(p1[1],p2[1])): #矩形2最高端大于矩形1最低端
        if(cross(p1,p2,p3)*cross(p1,p2,p4)<=0 and cross(p3,p4,p1)*cross(p3,p4,p2)<=0):
            D=1
        else:
            D=0
    else:
        D=0
    return D


#写入图片和Xml
def writePicAndXml(slide_x1, slide_y1, x_bSteps, y_bSteps, data):
    overlapRate = area_overlab(slide_x1, slide_y1, x_bSteps, y_bSteps, data)

    #保存图片
    dim = (600, 600)
    cropped = img[slide_y1 : slide_y1 + y_bSteps, slide_x1 : slide_x1 + x_bSteps]
    #resized = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)
    resized = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)
    #resized = transform.resize(cropped, (600, 600))
    imgPath_1 = saveAllImagePath + str(index).zfill(6) + '.jpg'
    #cv2.imwrite(imgPath_1,resized)
    cv2.imwrite(imgPath_1,resized)
    #写入VocXml

    while max(overlapRate) == 1:
        overlapRate.remove(1)

    ratio_x = x_bSteps / 600
    ratio_y = y_bSteps / 600


    #写入VocXml
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'sata'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = str(index).zfill(6) + '.jpg'

    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = 'SATALLITE'
    node_annotation = SubElement(node_source, 'annotation')
    node_annotation.text = 'sata'
    node_image = SubElement(node_source, 'image')
    node_image.text = 'sata'
    node_flickrid = SubElement(node_source, 'flickrid')
    node_flickrid.text = 'NULL'

    node_owner = SubElement(node_root, 'owner')
    node_flickrid = SubElement(node_owner, 'flickrid')
    node_flickrid.text = 'NULL'
    node_name = SubElement(node_owner, 'name')
    node_name.text = '联合视觉'

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '600'
    node_height = SubElement(node_size, 'height')
    node_height.text = '600'
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '0'

    for i in range(len(data)):
        x2 = data.values[i][0]
        y2 = data.values[i][1]
        w2 = data.values[i][2]
        h2 = data.values[i][3]

        xmin = 0
        ymin = 0
        xmax = 0
        ymax = 0

        ratio = area_overlab_double(slide_x1, slide_y1, x_bSteps, y_bSteps, x2, y2, w2, h2)
        if ratio == 1:
            xmin = int((x2 - slide_x1) / ratio_x)
            ymin = int((y2 - slide_y1) / ratio_y)
            xmax = int((x2 - slide_x1 + w2) / ratio_x)
            ymax = int((y2 - slide_y1 + h2) / ratio_y)
        elif ratio >= 0.35 and ratio < 1:
            if x2 < slide_x1:
                w2 = x2 + w2 - slide_x1
                x2 = slide_x1
            if y2 < slide_y1:
                h2 = y2 + h2 - slide_y1
                y2 = slide_y1
            if x2 + w2 > slide_x1 + x_bSteps:
                w2 = slide_x1 + x_bSteps - x2
            if y2 + h2 > slide_y1 + y_bSteps:
                h2 = slide_y1 + y_bSteps - y2

            xmin = int((x2 - slide_x1) / ratio_x)
            ymin = int((y2 - slide_y1) / ratio_y)
            xmax = int((x2 - slide_x1 + w2) / ratio_x)
            ymax = int((y2 - slide_y1 + h2) / ratio_y)
        else :
            continue


        #写入object1
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = 'ship'
        node_pose = SubElement(node_object, 'pose')
        node_pose.text = 'Unspecified'
        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(xmin)
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(ymin)
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(xmax)
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(ymax)

    xml = tostring(node_root)  #格式化显示
    dom = parseString(xml)

    #写入voc格式xml文件
    xml_name = os.path.join(saveAllVocXmlPath, str(index).zfill(6) + '.xml')
    with open(xml_name, 'wb') as f:
        f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))

    #写入残缺
    if max(overlapRate) > 0:
        while min(overlapRate) == 0:
            overlapRate.remove(0)
        #print('overlapRate', overlapRate)
        if not (min(overlapRate) > 0.95  or max(overlapRate) < 0.05):
            global defectIndex
            defectIndex += 1
            #cropped = img[slide_y1 : slide_y1 + y_bSteps, slide_x1 : slide_x1 + x_bSteps]
            imgPath_2 = saveDefectImagePath + str(index).zfill(6) + '.jpg'
            cv2.imwrite(imgPath_2,resized)

            #写入voc格式xml文件
            xml_name = os.path.join(saveDefectVocXmlPath, str(index).zfill(6) + '.xml')
            with open(xml_name, 'wb') as f:
                f.write(dom.toprettyxml(indent='\t', encoding='utf-8'))

            print('残缺率', overlapRate)

    print(str(index).zfill(6) + '.jpg')
    print('图片数量 = ', index, '图片残缺数量 = ', defectIndex, '裁切第'+str(picNumber)+'张图片')


#文件的开头
index = 1
global defectIndex
defectIndex = 0
bSteps = 800
downSteps = 800
picNumber = 0

x_bSteps = 800
y_bSteps = 800

slide_x1 = 0
slide_y1 = 0
# 扩展的步长
bStep = 10

datPath = 'g:/ship/船dat/'
imagePath = 'g:/ship/船tif/'
saveAllImagePath = 'd:/5_ship_gather/ship_pic_all/'
saveDefectImagePath = 'd:/5_ship_gather/ship_pic_incomplete/'
saveAllVocXmlPath = 'd:/5_ship_gather/ship_xml_all/'
saveDefectVocXmlPath = 'd:/5_ship_gather/ship_xml_incomplete/'

def xml_name(file_dir):
    for root, dirs, files in os.walk(imagePath):
#     #获取文件的名称 000001
    for file in files:
        line = line.strip('\n').split("_")[1].split(".")[0]
        print(line)

        try:
            print('正在裁切的tif图片为：', line)
            # iFile = file.split(".")[0]
            iFile = line
            #print(iFile)
            uImagePath = imagePath + iFile + '.tif'
            #xml读取操作，将获取到的xml文件名送入到dom中解析
            iDatPath = datPath + iFile + '.dat';
            if not os.path.exists(iDatPath):
                print(uImagePath, '不存在对应的.dat文件')
                continue

            #img = io.imread(uImagePath)
            img = cv2.imdecode(np.fromfile(uImagePath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            picNumber += 1
            print(img.shape)
            data = pd.read_csv(iDatPath,header=None,skiprows=[0],sep='\s+')
            # 在图片里滑动截图
            while (slide_y1 + bSteps < img.shape[0]):
                slide_x1 = 0
                while (slide_x1 + bSteps < img.shape[1]):
                    overlapRate = area_overlab(slide_x1, slide_y1, bSteps, bSteps, data)

                    # 对y做标记
                    before_slide_y = slide_y1

                    if(max(overlapRate) == 0):
                        slide_x1 += bSteps
                        continue

                    if(max(overlapRate) > 0 and max(overlapRate) < 1 / 2):
                        slide_x1 += bSteps
                        continue

                    if(max(overlapRate) > 1 / 2 and max(overlapRate) <= 1):
                        #检测到的都是残缺目标
                        # 判断左边
                        if(slide_x1 == 0):
                            # 每一列的第一个截图，不用考虑
                            a = 1
                        else:
                            #计算有没有在这条边重叠
                            differ_xl = xl_location(slide_x1, slide_y1, x_bSteps, y_bSteps, data, 1, bStep)
                            slide_x1 -= differ_xl
                            x_bSteps += differ_xl

                        # 判断上边
                        if(slide_y1 == 0):
                            # 每一行的截图，不用考虑
                            a = 1
                        else:
                            #计算有没有在这条边重叠
                            differ_yt = xl_location(slide_x1, slide_y1, x_bSteps, y_bSteps, data, 2, bStep)
                            slide_y1 -= differ_yt
                            y_bSteps += differ_yt

                        # 判断右边
                        #计算有没有在这条边重叠
                        differ_xr = xl_location(slide_x1, slide_y1, x_bSteps, y_bSteps, data, 3, bStep)
                        x_bSteps += differ_xr

                        # 判断下边
                        #计算有没有在这条边重叠
                        differ_yb = xl_location(slide_x1, slide_y1, x_bSteps, y_bSteps, data, 4, bStep)
                        y_bSteps += differ_yb

                        #print(differ_xl, differ_yt, differ_xr, differ_yb)



                    # 判断数量
                    getBoxNums, flag_4= area_overlab_compare_dat(slide_x1, slide_y1, x_bSteps, y_bSteps, data)
                    # if len(getBoxNums) > 3 and max(getBoxNums) > 0.8 and flag_4 == true:
                    print('getBoxNums', getBoxNums)
                    while min(getBoxNums) == 0:
                        getBoxNums.remove(0)

                    if min(getBoxNums) > 0.5 and flag_4 == 1:
                        writePicAndXml(slide_x1, slide_y1, x_bSteps, y_bSteps, data)
                        index += 1

                    #初始化参数
                    # index += 1
                    x_bSteps = 800
                    y_bSteps = 800
                    slide_x1 += x_bSteps

                    # 让y回归初始位置
                    slide_y1 = before_slide_y

                    print('----------------------------')

                slide_y1 += downSteps

            #每张图片读取完，初始化参数
            slide_y1 = 0
            slide_x1 = 0
            cv2.waitKey (0)
            cv2.destroyAllWindows()
        except:
            print("not .dat文件")


