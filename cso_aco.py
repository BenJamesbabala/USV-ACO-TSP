# -*- coding: utf-8 -*-
import random
import sys
import numpy as np

# ----------- 蚂蚁算法 -----------

# 参数
# (ALPHA, BETA, RHO, Q) = (1.0, 1.0, 0.6, 30.0)
(alpha,beta,rho,smp_max,smp_min,srd,K,Q)=(1.0,1.0,0.6,4,1,0.3,5,3.0)
def change(array,i,n):
    temp = np.random.randint(0,n-2)
    
    if i != temp:
        p = array[temp]
        array[temp] = array[i];
        array[i] = p;
    return array

def seeking(city_num,distance_graph,Route_l,K,smp_min,smp_max):
    smp = np.zeros([K,city_num])
    for i in range(K):
        smp[i,:] = Route_l[:]
    smp = smp.astype(int)
    for i in range(K-1):
        srdnum = (smp_min + np.random.random()* (smp_max - smp_min))//1

        count = 0
        for j in range(city_num-2):
            if np.random.random() <=srd:
                smp[i,1:city_num-1] = change(smp[i,1:city_num-1],j,city_num)
                count = count +1
                if count >= srdnum:
                    break

    f = np.zeros([K])
    for i in range(K):
        for j in range(city_num-1):
            f[i] = f[i] + distance_graph[smp[i,j],smp[i,j+1]]
    mindex = np.argmin(f)

    Route = smp[mindex,:]


    return Route

class Ant_cso(object):
    # 初始化
    def __init__(self,cityNum,distance_graph,iter_max,ant_num,start_city,end_city):

        self.city_num=cityNum  # 城市数目
        self.distance_graph=np.array(distance_graph)  #距离矩阵
        self.iter_max = iter_max
        self.ant_num = ant_num
        self.start_city = start_city
        self.end_city = end_city

        self.pheromone_graph=np.ones([self.city_num,self.city_num])    #信息素矩阵
        self.Table = np.zeros([ant_num,cityNum])
        self.Length = np.zeros([self.ant_num,1])

        self.Route_best_l = np.zeros([iter_max,self.city_num])
        self.Length_best_l = np.zeros([iter_max,1])
        self.Length_ave = np.zeros([iter_max,1])
        self.citys_index = np.arange(cityNum)
        

    def _aco(self):     #一次蚁群算法的遍历  
        start = np.zeros([self.ant_num,1])
        for i in range(self.ant_num):
            start[i] = int(self.start_city)

        self.Table[:,0] = start.ravel()
        for i in range(self.ant_num):
            for j in range(2,self.city_num+1):
                tabu = self.Table[i,0:(j-1)]

                allow_index =  np.array(list(set(self.citys_index).difference(set(tabu))))

                if allow_index.shape[0]>1:
                    allow_index=np.array(list(set(allow_index).difference([self.end_city])))
                P = np.array(allow_index).astype(float)
                for k in range(allow_index.shape[0]):
                    P[k]=pow(self.pheromone_graph[int(tabu[-1]),allow_index[k]],alpha)*pow((1/self.distance_graph[int(tabu[-1]),allow_index[k]]),beta)


                P = (1/np.sum(P))*P
                

                Pc = P.cumsum()
                
                target_index = np.where(Pc>np.random.random(1))
                
                target = allow_index[target_index][0]

                self.Table[i,j-1]=target

    def Route_length(self):
        self.Length = np.zeros([self.ant_num,1])
        for i in range(self.ant_num):
            Route = self.Table[i,:].astype(int)
            for j in range(self.city_num-1):
                self.Length[i] = self.Length[i]+self.distance_graph[Route[j],Route[j+1]]
                   
    
    def _cso(self):
        L2 = np.sort(self.Length,axis=None)

        Route_l = np.zeros([1,self.city_num])
        for i in range(K):
            seeking_index = np.where(self.Length==L2[i])
            
            if seeking_index[0].shape != 1:
                seeking_index = seeking_index[0]
            
            Route_l = self.Table[seeking_index[0],:]
            Route_l = seeking(self.city_num,self.distance_graph,Route_l,K,smp_min,smp_max)
            self.Table[seeking_index,:] = Route_l

    def Clean_update(self):
        Delta_pheromone_graph = np.zeros([self.city_num,self.city_num])
        
        self.Table.astype(int)
        for i in range(self.ant_num):
            for j in range(self.city_num-1):
                Delta_pheromone_graph[int(self.Table[i,j]),int(self.Table[i,j+1])] = Delta_pheromone_graph[int(self.Table[i,j]),int(self.Table[i,j+1])]+Q/self.Length[i]
        self.pheromone_graph = (1-rho)*self.pheromone_graph+Delta_pheromone_graph
        self.Table = np.zeros([self.ant_num,self.city_num])

   

    # 搜索路径，外部接口
    def search_path(self):
        for iter in range(self.iter_max):
            self._aco()
            self.Route_length()
            self._cso()
            
            min_index = np.argmin(self.Length)
            min_Length = self.Length[min_index]
            self.Length_best_l[iter] = min_Length
            self.Length_ave[iter] = np.mean(self.Length)
            
            self.Route_best_l[iter,:] = self.Table[min_index,:]

            self.Clean_update()
            
       