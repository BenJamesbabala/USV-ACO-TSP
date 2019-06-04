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
import math

import PathPlaningStrategy as ps

import EviStruct as ES
from EviStruct import Point
from EviStruct import OffsetCoord
from EviStruct import SquareProperty
from EviStruct import square_directions
from EviStruct import square_Neighbordirections

import cso_aco

f = figure(figsize=(7,7))
ax = plt.subplot(111)
squaresize = 0.002*1.6118548977
ullong=-170.57
ullati=53.18
#下右
drlong=-169.24
drlati=52.60
m = Basemap(projection='merc', llcrnrlat=drlati, urcrnrlat=ullati, \
            llcrnrlon=(ullong + 0.002), urcrnrlon=(drlong - 0.002), lat_ts=1, resolution='i')
m.drawcountries(linewidth=0.1, color='r')


# 从Txt中读取环境建模保存的信息
squareprolist = []
print('Start to read Evi information')
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
print('Read done')




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


diagram4 = ps.GridWithWeights(Navigonal)
# 加入不可航性区域
diagram4.walls = DisNavigonal
# 加入每一个点的权重
diagram4.weights = Naviweight


Point0 = ps.get_polygonIndexfromPoint(-170.229145,52.701435,squareprolist)
# 目标点
Point9 = ps.get_polygonIndexfromPoint(-169.791635,52.895138,squareprolist)
Point1 = ps.get_polygonIndexfromPoint(-170.094576,52.786191,squareprolist)
# 目标点
Point2 = ps.get_polygonIndexfromPoint(-170.033261,52.740627,squareprolist)
Point3 = ps.get_polygonIndexfromPoint(-169.845021,52.782513,squareprolist)
# 目标点
Point4 = ps.get_polygonIndexfromPoint(-169.775323,52.755082,squareprolist)
Point5 = ps.get_polygonIndexfromPoint(-169.627611,52.827083,squareprolist)
# 目标点
Point6 = ps.get_polygonIndexfromPoint(-169.705849,52.89450,squareprolist)
Point7 = ps.get_polygonIndexfromPoint(-169.909834,52.867746,squareprolist)
# 目标点
Point8 = ps.get_polygonIndexfromPoint(-170.009920,52.852899,squareprolist)

Point_l_m = [Point0,Point1,Point2,Point3,Point4,Point5,Point6,Point7,Point8,Point9]

def getLineLength(firstwayPnt, secondwayPnt):
    x = [firstwayPnt.y, secondwayPnt.y]
    y = [firstwayPnt.x, secondwayPnt.x]
    # compute map projection coordinates for lat/lon grid.
    xlist, ylist = m(y, x)  # 将经纬度转换为图片所用坐标,米坐标
    return np.sqrt((xlist[0] - xlist[1]) * (xlist[0] - xlist[1]) + (ylist[0] - ylist[1]) * (ylist[0] - ylist[1]))  # 米坐标

Point_l = []
for iter in Point_l_m:
    Point_l.append(ES.getCenterPoint(ullong, ullati, squaresize,iter) )


(city_num, ant_num,iter_max) = (10,25,300)

distance_graph = [[0.0 for col in range(city_num)] for raw in range(city_num)]  # 距离矩阵，默认为0
distance_graph_update = [[0 for col in range(city_num)] for raw in range(city_num)]  # 距离矩阵，默认为0
distance_graph_update = np.array(distance_graph_update)
distance_graph = np.array(distance_graph)

for i in range(city_num):  #xrange用法与range相同，所不同的是生成的不是一个数组，而是一个生成器。
            for j in range(city_num):
                distance_graph[i][j]=getLineLength(Point_l[i],Point_l[j])

Route_point_list = []
Route_l = []

while ((len(Route_l) <= 2) or ((Route_l[-1] != Route_l[-2]).any())):
    
    cso_aco_ = cso_aco.Ant_cso(city_num,distance_graph,iter_max,ant_num,0,8)
    cso_aco_.search_path()

    index = np.argmin(cso_aco_.Length_best_l)
    Route = cso_aco_.Route_best_l[index,:].astype(int)
    Route_l.append(Route)
   
    print('Route:',len(Route_point_list),'Route:',Route,\
          not((len(Route_l) <= 2) or ((Route_l[-1] != Route_l[-2]).any())))

    for i in range(Route.shape[0]-1):
        if distance_graph_update[Route[i],Route[i+1]] == 0:
            came_from, cost_so_far, Iteracount = ps.a_star_search(diagram4, Point_l_m[Route[i]], Point_l_m[Route[i+1]])
            pathFinderPoints = []
            
            # 回溯父节点
            pathFinderPoints.append(Point_l_m[Route[i+1]])
            fatherNode = came_from[Point_l_m[Route[i+1]]]
            while fatherNode != Point_l_m[Route[i]]:
                pathFinderPoints.append(fatherNode)
                fatherNode = came_from[fatherNode]
            
            pathFinderPoints.append(Point_l_m[Route[i]])#所有的网格节点信息

            Route_point_list.append(pathFinderPoints)
            print(len(Route_point_list))
            planeddistanceall = 0
            for j in range(len(pathFinderPoints) - 1):
                planeddistanceall = planeddistanceall + getLineLength(ES.getCenterPoint(ullong, ullati, squaresize, pathFinderPoints[i]),ES.getCenterPoint(ullong, ullati, squaresize, pathFinderPoints[i+1]) )  # 米坐标

            distance_graph[Route[i],Route[i+1]] = planeddistanceall
            distance_graph[Route[i+1],Route[i]] = planeddistanceall
            

            distance_graph_update[Route[i],Route[i+1]]=len(Route_point_list)
            distance_graph_update[Route[i+1],Route[i]]=len(Route_point_list)
    

print('list',Route_l)
print('best',Route)

print('绘制环境地图')
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

print('绘制航线')
print(Route)
for i in range(9):
    
    print(Route[i],Route[i+1])
    print(distance_graph_update[Route[i],Route[i+1]]-1)
    pathFinderPoints = Route_point_list[distance_graph_update[Route[i],Route[i+1]]-1]
    lineCount = 0
    PlanedcenterPnt=[]
    while lineCount < (len(pathFinderPoints) - 1):
        last = pathFinderPoints[lineCount]
        new = pathFinderPoints[lineCount + 1]
        
        lastCenter = ES.getCenterPoint(ullong, ullati, squaresize, last)  # Point
        PlanedcenterPnt.append(lastCenter)
        newCenter = ES.getCenterPoint(ullong, ullati, squaresize, new)  # Point
        if lineCount == 0:
            y1 = [lastCenter.y]
            x1 = [lastCenter.x]
            x1, y1 = m(x1, y1)
            x1 = x1[0]
            y1 = y1[0]
            plt.scatter(x1, y1, c='r', marker='o')
#             plt.text(x1,y1,'0',fontdict={'size':20,'color':'g'})
        if lineCount == len(pathFinderPoints) - 2:
            y1 = [newCenter.y]
            x1 = [newCenter.x]
            x1, y1 = m(x1, y1)
            m.plot(x1, y1, c='r', marker='o')
            
        x = [lastCenter.y, newCenter.y]
        y = [lastCenter.x, newCenter.x]
        xlist, ylist = m(y, x)  # 将经纬度转换为图片所用坐标
        m.plot(xlist, ylist, color='b', linestyle=":")  # 首先画白线
        m.plot(xlist, ylist, color='r')  # 首先画白线
        lineCount = lineCount + 1
j = 0
for iter in Point_l:
    x,y = m(iter.x,iter.y)
    plt.text(x,y,j,fontdict={'size':20,'color':'g'})
    j= j+1

plt.show()

