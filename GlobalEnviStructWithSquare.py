"""
Copyright(c) 2017, waylon
All rights reserved.
Distributed under the BSD license.
"""

# -*- coding: utf-8 -*-

# 全局环境建模

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import sys
from pylab import figure
from mpl_toolkits.basemap import Basemap
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from shapely.geometry import Polygon  # 比较多边形交叉
import collections
import heapq  # heapq是一种子节点和父节点排序的树形数据结构
from math import isnan
from collections import Iterable
import json
import copy

import EviStruct
from EviStruct import Point
from EviStruct import OffsetCoord
from EviStruct import SquareProperty
from EviStruct import square_directions
from EviStruct import square_Neighbordirections


Point = collections.namedtuple("Point", ["x", "y"])
OffsetCoord = collections.namedtuple("OffsetCoord", ["row", "col"])
SquareProperty = collections.namedtuple("SquareProperty",
                                        ["offsetCoord", "centerPoint", "cornerPoints", "weight", "squareSize",
                                         "isNavigonal"])
square_directions = [OffsetCoord(1, 1), OffsetCoord(1, -1), OffsetCoord(-1, -1), OffsetCoord(-1, 1)]
square_Neighbordirections = [OffsetCoord(0, -1), OffsetCoord(0, 1), OffsetCoord(-1, 0), OffsetCoord(1, 0),
                             OffsetCoord(1, 1), OffsetCoord(1, -1), OffsetCoord(-1, -1), OffsetCoord(-1, 1)]


# 得到某区域正方形网格的所有属性
def SquarePro(ullong, ullati, squaresize, squareColumn, squareRow):
    savesquareprolist = []
    rowCount = 0
    while rowCount < squareRow:
        centerY = ullati - (rowCount + 0.5) * squaresize  # 中心点纬度
        columnCount = 0
        while columnCount < squareColumn:
            centerX = ullong + (columnCount + 0.5) * squaresize  # 中心点经度
            cornerlist = GetSquareCorners(centerX, centerY, squaresize)
            squarepro = SquareProperty(OffsetCoord(rowCount, columnCount), Point(centerX, centerY), cornerlist, 1,
                                       squaresize, True)
            savesquareprolist.append(squarepro)
            columnCount = columnCount + 1
        rowCount = rowCount + 1
    return savesquareprolist


def intersection_recognition(singleCornerslist, lons, lats):
    p1 = Polygon([(singleCornerslist[0].x, singleCornerslist[0].y), (singleCornerslist[1].x, singleCornerslist[1].y),
                  (singleCornerslist[2].x, singleCornerslist[2].y), (singleCornerslist[3].x, singleCornerslist[3].y)])
    i = 0
    polygonlist = []
    while i < len(lats):
        t = (lons[i], lats[i])
        polygonlist.append(t)
        i = i + 1
    p2 = Polygon(polygonlist)
    return p1.intersects(p2)

# 得到某方向的一个点
def GetCorner(centerX, centerY, squaresize, cornerCount):
    cornerX = centerX + squaresize / 2 * square_directions[cornerCount].row
    cornerY = centerY + squaresize / 2 * square_directions[cornerCount].col
    return Point(cornerX, cornerY)

def getCenterPoint(ullong, ullati, squaresize, offsetcoord):
    rowCount = offsetcoord.row
    columnCount = offsetcoord.col
    centerY = ullati - (rowCount + 0.5) * squaresize  # 中心点纬度
    centerX = ullong + (columnCount + 0.5) * squaresize  # 中心点经度
    return Point(centerX, centerY)

# 得到一个正方形四个点的坐标
def GetSquareCorners(centerX, centerY, squaresize):
    cornerSum = 4
    cornerCount = 0
    cornerPoints = []
    while cornerCount < cornerSum:
        Points = GetCorner(centerX, centerY, squaresize, cornerCount)
        cornerPoints.append(Points)
        cornerCount = cornerCount + 1
    return cornerPoints




def getweight(squareCoord, DisNavigonal):
    results = [square_neighbor(squareCoord, 0), square_neighbor(squareCoord, 1), square_neighbor(squareCoord, 2), \
               square_neighbor(squareCoord, 3), square_neighbor(squareCoord, 4), square_neighbor(squareCoord, 5), \
               square_neighbor(squareCoord, 6), square_neighbor(squareCoord, 7)]
    neighbourDisnaCount = 0;
    for x in results:
        if (x in DisNavigonal):
            neighbourDisnaCount += 1
    if (neighbourDisnaCount > 0):
        return 0.5 * (2 ** (neighbourDisnaCount - 1))
    else:
        return 0

def square_neighbor(squareoffset, direction):
    # print(hex_add(hexcube, hex_direction(direction)))
    return square_add(squareoffset, square_Neighbordirections[direction])


def square_add(a, b):
    return OffsetCoord(a.row + b.row, a.col + b.col)



squaresize = 0.002*1.6118548977
# 创建地图
f = figure(figsize=(15, 15))
ax = plt.subplot(111)
# plt.ion()  #interactive mode on   交互模式
ullong=-170.57
ullati=53.18
#下右
drlong=-169.24
drlati=52.60
m = Basemap(projection='merc', llcrnrlat=drlati, urcrnrlat=ullati, \
            llcrnrlon=(ullong + 0.002), urcrnrlon=(drlong - 0.002), lat_ts=1, resolution='i')
m.drawcountries(linewidth=0.1, color='r')
# 正方形行数和列数
# print(drlong-ullong)
squareColumn = (drlong - ullong) // squaresize + 1  # 列数
squareRow = (ullati - drlati) // squaresize + 1  # 行数
print("网格化后矩阵维数 %d * %d" % (squareRow, squareColumn))
# 获得指定大小的正方形网格

squareprolist = EviStruct.SquarePro(ullong, ullati, squaresize, squareColumn, squareRow)#空白的网格地图

try:
    # 加载XML文件（2种方法,一是加载指定字符串，二是加载指定文件）
    tree = ET.parse("./data/US4AK62M.xml")  # 打开xml文档
    print('成功读取')
    # tree= ElementTree.fromstring(text)
    root = tree.getroot()  # 获得root节点
except Exception as e:
    print("Error:cannot parse file")
    sys.exit(1)
for layerInfo in root.findall("LayerInfo"):  # 找到root节点下的所有xx节点
    for layer in layerInfo.findall("Layer"):  # 找到root节点下的所有xx节点
        # print(layer.get('id'))
        # print(layer.find('layerName').text)
        for feature in layer.find("Features"):
            feaid = feature.get('id')  # 子节点下属性name的值
            # print(feaid)
            elevation = feature.find('valdco').text#海拔
            # print(elevation)
            for wayPoints in feature.findall("wayPoints"):
                lats = []
                lons = []
                # high=[]
                for waypoint in wayPoints.findall("waypoint"):
                    waypoint_ID = waypoint.find('id').text
                    longitude = waypoint.find('lon').text
                    latitude = waypoint.find('lat').text
                    lons.append(float(longitude))
                    lats.append(float(latitude))
                # 两个多边形交叉判断
                if (len(lats) > 2):
                    compareCount = 0
                    while compareCount < len(squareprolist):
                        if (squareprolist[compareCount].isNavigonal == True):  # 可航时才检查
                            singleCornerslist = squareprolist[compareCount].cornerPoints
                            ans = EviStruct.intersection_recognition(singleCornerslist, lons, lats)  # 两个多边形交叉返回true，否则false
                            if (ans):
                                # 标记不可航行区域
                                temp = squareprolist[int(compareCount)]._replace(isNavigonal=False)
                                # print("不可航行")
                                squareprolist[int(compareCount)] = temp
                        compareCount = compareCount + 1
print("读取完成")
# 设置权重
i = 0
Navigonal = []
DisNavigonal = []

while i < len(squareprolist):
    Navigonal.append((squareprolist[i].offsetCoord))
    if (squareprolist[i].isNavigonal == False):
        DisNavigonal.append((squareprolist[i].offsetCoord))
    i = i + 1
# 权重
Naviweight = {}  # 字典
weightIndex = 0
while weightIndex < len(squareprolist):
    weightoffset = 1+EviStruct.getweight(squareprolist[weightIndex].offsetCoord, DisNavigonal)
    temp = squareprolist[int(weightIndex)]._replace(weight=weightoffset)
    squareprolist[int(weightIndex)] = temp
    weightIndex += 1

# 环境建模完成，需要保存到txt中
# 写入到txt中
with open("./data/US4AK62M.txt", "w") as f:
    for square in squareprolist:
        f.write(json.dumps(square._asdict()))
        f.write("\n")
print("环境建模完成")

#绘制正方形网格以及可航区域与不可航区域
plotsquareCount=0
while plotsquareCount<len(squareprolist):
    squarecorner= squareprolist[plotsquareCount].cornerPoints
    shadowlons=[squarecorner[0].x,squarecorner[1].x,squarecorner[2].x,squarecorner[3].x]
    shadowlats=[squarecorner[0].y,squarecorner[1].y,squarecorner[2].y,squarecorner[3].y]
    x, y = m(shadowlons, shadowlats)
    shpsegs = []
    shpsegs.append(list(zip(x,y)))
    lines = LineCollection(shpsegs,antialiaseds=(1,))
    #不可航行区域
    if(squareprolist[plotsquareCount].isNavigonal==False):
        lines.set_facecolors(cm.jet(0.1))
        lines.set_edgecolors('g')
        lines.set_linewidth(0.6)
        lines.set_alpha(0.6) #设置透明度
        ax.add_collection(lines)  #绘制不可行区域
    else:
        lines.set_facecolors(cm.jet(0.02))
        lines.set_edgecolors('b')
        lines.set_linewidth(1.2)
        #lines.set_alpha(0.1) #设置透明度

        # 设置颜色深度，权重越大深度越大
        weight = Naviweight.get(squareprolist[plotsquareCount].offsetCoord, 1)
        if weight < 1:
            weight = 1
        if weight > 10:
            weight = 10
        lines.set_alpha(0.1 * weight)  # 设置透明度
        # ax.add_collection(lines)
    plotsquareCount=plotsquareCount+1
print("环境构建成功")




plt.show()