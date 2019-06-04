
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
import GeometryBase

import PathPlaningStrategy as ps

import EviStruct as ES
from EviStruct import Point
from EviStruct import OffsetCoord
from EviStruct import SquareProperty
from EviStruct import square_directions
from EviStruct import square_Neighbordirections


def getLineLength(firstwayPnt, secondwayPnt):
    x = [firstwayPnt.y, secondwayPnt.y]
    y = [firstwayPnt.x, secondwayPnt.x]
    # compute map projection coordinates for lat/lon grid.
    xlist, ylist = m(y, x)  # 将经纬度转换为图片所用坐标,米坐标
    return np.sqrt((xlist[0] - xlist[1]) * (xlist[0] - xlist[1]) + (ylist[0] - ylist[1]) * (ylist[0] - ylist[1]))  # 米坐标


# 创建地图
f = figure(figsize=(10,10))
ax = plt.subplot(111)
# square size
squaresize = 0.0015
ullong=-170.57
ullati=53.18
#下右
drlong=-169.24
drlati=52.60
m = Basemap(projection='merc', llcrnrlat=drlati, urcrnrlat=ullati, \
            llcrnrlon=(ullong + 0.002), urcrnrlon=(drlong - 0.002), lat_ts=1, resolution='i')
m.drawcountries(linewidth=0.1, color='r')
#绘制从电子海图解析出的陆地等不可航区域

# 从Txt中读取环境建模保存的信息
squareprolist = []
print('Start to read txt')
with open("./data/US4AK62M.txt", "r") as r:
    filelist = r.readlines()
    for x in filelist:
        dic = json.loads(x)  # 输出dict类型
        p = SquareProperty(**dic)
        offsetCoord=OffsetCoord(p.offsetCoord[0],p.offsetCoord[1])
        centerPoint=Point(p.centerPoint[0],p.centerPoint[1])
        cornerPoints=[]
        point1=Point(p.cornerPoints[0][0],p.cornerPoints[0][1])
        cornerPoints.append(point1)
        point2=Point(p.cornerPoints[1][0],p.cornerPoints[1][1])
        cornerPoints.append(point2)
        point3=Point(p.cornerPoints[2][0],p.cornerPoints[2][1])
        cornerPoints.append(point3)
        point4=Point(p.cornerPoints[3][0],p.cornerPoints[3][1])
        cornerPoints.append(point4)
        isnavigonal=p.isNavigonal
        squaresize=p.squareSize
        weight=p.weight
        squarepro=SquareProperty(offsetCoord,centerPoint,cornerPoints,weight,squaresize,isnavigonal)
        squareprolist.append(squarepro)

# 输出航行区域与不可航行区域
i = 0
Navigonal = []
DisNavigonal = []
# 权重
Naviweight = {}  # 字典
while i < len(squareprolist):
    Navigonal.append((squareprolist[i].offsetCoord))
    Naviweight[squareprolist[i].offsetCoord] = squareprolist[i].weight
    if (squareprolist[i].isNavigonal == False):
        DisNavigonal.append((squareprolist[i].offsetCoord))
    i = i + 1

#绘制正方形网格以及可航区域与不可航区域
print('Start to plot')
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
        ax.add_collection(lines)
    plotsquareCount=plotsquareCount+1
print("环境构建成功")
x,y = m(-170.229145,52.701435)
m.scatter(x,y, c='r', marker='o')
# -*- coding: utf-8 -*-
"""
Copyright(c) 2017, waylon
All rights reserved.
Distributed under the BSD license.
"""
plt.show()



diagram4 = ps.GridWithWeights(Navigonal)
# 加入不可航性区域
diagram4.walls = DisNavigonal
# 加入每一个点的权重
diagram4.weights = Naviweight
# 起始点

startPoint = ps.get_polygonIndexfromPoint(-170.54, 52.70,squareprolist)#获取点的方格坐标
print(startPoint)
# 目标点
targetPoint = ps.get_polygonIndexfromPoint(-169.50, 53.10,squareprolist)
print(targetPoint)
# came_from, cost_so_far = dijkstra_search(diagram4,startPoint,targetPoint)
came_from, cost_so_far, Iteracount = ps.a_star_search(diagram4, startPoint, targetPoint)
# print("遍历点数 %d" % (Iteracount))
print("A*算法寻得最短路径")
pathFinderPoints = []
# 回溯父节点
pathFinderPoints.append(targetPoint)
fatherNode = came_from[targetPoint]
while fatherNode != startPoint:
    pathFinderPoints.append(fatherNode)
    fatherNode = came_from[fatherNode]
    
pathFinderPoints.append(startPoint)#所有的网格节点信息
print("优化前节点个数： %d" % len(pathFinderPoints))
# 在地图上绘制最短路径
lineCount = 0;
PlanedcenterPnt=[]
while lineCount < (len(pathFinderPoints) - 1):
    last = pathFinderPoints[lineCount]
    new = pathFinderPoints[lineCount + 1]
    # print(last)
    lastCenter = ES.getCenterPoint(ullong, ullati, squaresize, last)  # Point
    PlanedcenterPnt.append(lastCenter)
    newCenter = ES.getCenterPoint(ullong, ullati, squaresize, new)  # Point
    if lineCount == 0:
        y1 = [lastCenter.y]
        x1 = [lastCenter.x]
        x1, y1 = m(x1, y1)
        m.scatter(x1, y1, c='r', marker='o')
    if lineCount == len(pathFinderPoints) - 2:
        y1 = [newCenter.y]
        x1 = [newCenter.x]
        x1, y1 = m(x1, y1)
        m.plot(x1, y1, c='r', marker='o')
    x = [lastCenter.y, newCenter.y]
    y = [lastCenter.x, newCenter.x]
    xlist, ylist = m(y, x)  # 将经纬度转换为图片所用坐标
    m.plot(xlist, ylist, color='b', linestyle=":")  # 首先画白线
    m.plot(xlist, ylist, color='g')  # 首先画白线
    lineCount = lineCount + 1
last = pathFinderPoints[lineCount]
lastCenter = ES.getCenterPoint(ullong, ullati, squaresize, last)  # Point
PlanedcenterPnt.append(lastCenter)
# 规划路径曲线长度
wayCount = 0
planeddistanceall = 0
while wayCount < (len(PlanedcenterPnt) - 1):
    planeddistanceall = planeddistanceall + getLineLength(PlanedcenterPnt[wayCount], PlanedcenterPnt[wayCount + 1])  # 米坐标
    wayCount += 1
print(planeddistanceall)

# A*曲线优化为近似直线
curvelist = copy.deepcopy(pathFinderPoints)
curvelist.reverse()  # 逆序排列
curveCount = 0
distance = 1.5  # 设置距离，但是这里不太有效吧，仍然有问题
while curveCount < len(curvelist) - 2:
    # print("优化中")
    # 检测是否可以优化
    curvelist = ps.curveoptimize(curveCount, curvelist, DisNavigonal, distance)
    curveCount += 1;
print("路径曲线优化结束")
# 曲线再次优化，算法存在问题，之前不能优化的，现在又可以优化了
curveCount = 0;
distance = 2
while curveCount < len(curvelist) - 2:
    # print("优化中")
    # 检测是否可以优化
    curvelist = ps.curveoptimize(curveCount, curvelist, DisNavigonal, distance)
    curveCount += 1;
print("路径曲线再次优化结束")
# 得到节点中心点和个数
print(curvelist)
centerPnt = []
print("优化后节点个数： %d" % len(curvelist))
for node in curvelist:
    for waypnt in squareprolist:
        if waypnt.offsetCoord == node:
            centerPnt.append(waypnt.centerPoint)
            break
print(centerPnt)
# 规划路径曲线长度
wayCount = 0
distanceall = 0
while wayCount < (len(centerPnt) - 1):
    distanceall = distanceall + getLineLength(centerPnt[wayCount], centerPnt[wayCount + 1])  # 米坐标
    wayCount += 1
print(distanceall)
lineCount = 0
while lineCount < (len(curvelist) - 1):
    last = curvelist[lineCount]
    new = curvelist[lineCount + 1]
    lastCenter = ES.getCenterPoint(ullong, ullati, squaresize, last)  # Point
    newCenter = ES.getCenterPoint(ullong, ullati, squaresize, new)  # Point
    x = [lastCenter.y, newCenter.y]
    y = [lastCenter.x, newCenter.x]
    xlist, ylist = m(y, x)  # 将经纬度转换为图片所用坐标
    m.plot(xlist, ylist, color='r')  # 首先画白线
    lineCount = lineCount + 1
plt.show()
print("界面显示成功")
