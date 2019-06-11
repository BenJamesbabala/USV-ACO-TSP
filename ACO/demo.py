import random
import copy
import sys
import tkinter as tk
import threading
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import AntColonyCore
import math
from functools import reduce


(city_num, ant_num,iter_max) = (30,50,100)

distance_x=[]
distance_x.append(10)
for x in range(city_num-1):
    distance_x.append(int((random.random()*700+50)//1))
distance_y=[]
distance_y.append(10)
for x in range(city_num-1):
    distance_y.append(int((random.random()*500+50)//1))




distance_graph = [[0.0 for col in range(city_num)] for raw in range(city_num)]  # 距离矩阵，默认为0
pheromone_graph = [[1.0 for col in range(city_num)] for raw in range(city_num)] #信息素矩阵，默认为1

iterDistance=[]  # 缓存距离数值，用于最后的性能分析


def DistancebyEuclidean(x1, y1,x2, y2):
    temp_distance = pow((x1 - x2), 2) + pow((y1 - y2), 2)
    return math.sqrt(temp_distance)



for i in range(city_num):  #xrange用法与range相同，所不同的是生成的不是一个数组，而是一个生成器。
            for j in range(city_num):
                distance_graph[i][j]=DistancebyEuclidean(distance_x[i],distance_y[i],distance_x[j],distance_y[j])


def __update_pheromone_gragh():

        # 获取每只蚂蚁在其路径上留下的信息素
        temp_pheromone = [[0.0 for col in range(city_num)] for raw in range(city_num)]
        for ant in ants:
            for i in range(1, city_num):
                start, end = ant.path[i - 1], ant.path[i]
                # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                temp_pheromone[start][end] += AntColonyCore.Q / ant.total_distance  # 信息素
                temp_pheromone[end][start] = temp_pheromone[start][end]

        # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = pheromone_graph[i][j] * AntColonyCore.RHO + temp_pheromone[i][j]  # 上一次的信息素*衰减因子+此次的信息素


for i in range(city_num):
    for j in range(city_num):
        pheromone_graph[i][j] = 1.0

ants = [AntColonyCore.Ant(ID,city_num,distance_graph,pheromone_graph) for ID in range(ant_num)]  # 初始蚁群
best_ant = AntColonyCore.Ant(-1,city_num,distance_graph,pheromone_graph)  # 初始最优解
best_ant.total_distance = 1 << 31  # 初始最大距离
iter = 1  # 初始化迭代次数


for i in range(iter_max):  #迭代次数
    # 遍历每一只蚂蚁
    for ant in ants:
        # 搜索一条路径
        ant.search_path()
        #ant.__update_pheromone_gragh()
        # 与当前最优蚂蚁比较
        if ant.total_distance < best_ant.total_distance:
            # 更新最优解
            best_ant = copy.deepcopy(ant)
    #记录此次迭代最短距离
    iterDistance.append(best_ant.total_distance)
    # 更新信息素
    # 所有蚂蚁完成对城市的遍历后，更新信息素
    __update_pheromone_gragh()
    iter = i

    print(u"迭代次数：", iter, u"最佳路径总距离：", int(best_ant.total_distance))
print(u"最优路径:",best_ant.path)
