# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:18:07 2019

@author: Prince Raj Roy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from scipy.cluster.hierarchy import dendrogram
from numba import jit
import time

'''df = pd.DataFrame({
    'x': [1, 1.5, 5, 3, 4, 3],
    'y': [1, 1.5, 5, 4, 4, 3.5]
    })'''

df1 = pd.read_csv("t4.8k.dat",delimiter=' ', names = ['x', 'y'])
df = df1.sample(n=20).reset_index(drop=True)
#print(df)
#df['y'] = list(df1['y'].sample(n=100, random_state=1))

#dendro = ff.create_dendrogram(df)
#dendro['layout'].update({'width':800, 'height':500})
#py.iplot(dendro, filename='simple_dendrogram')

colors = {i :"#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(df))}

dist = np.reshape(np.zeros((len(df),len(df))), (len(df),len(df)))
dist = pd.DataFrame(dist, columns=list(range(len(df))), index=list(range(len(df))))

intermediate_arrays = []
abc = []
added = []
done = {}
yes = 0
Z = []
length = len(df)


@jit(nopython=True)
def euclidean(x, y, x1, y1):
    return np.round(np.sqrt((x - x1) ** 2 + (y - y1) ** 2), 2)

@jit
def create_distance_matrix():
    for index, x, y in df.itertuples():
        for index1, x1, y1 in df.itertuples():
            dist[index][index1] = euclidean(x,y,x1,y1)
            dist[index1][index] = dist[index][index1]


def update_distance_matrix():
    global length
    global Z
    global done
    x=1
    y=1
    no = 0
    key = 0
    small = max(dist.max())+1
    for i in dist.index:
        for j in dist.columns:
            if i != j:
                if dist[i][j] < small:
                    small = dist[i][j]
                    x = i
                    y = j
    
    #For The Dendrogram-----------------------------------------
    b = []
    if len(dist) != 2:
        for p in done.keys():
            if x in done[p] or y in done[p]:
                no = 1
                key = p
                b.append(p)

        if no:
            done[length] = []
            l = y if x in done[key] else x
            done[length].append(l)
            done[length].extend(done[key][:-1])
            count = done[key][-1]['count']+1
            
            temp = []
            a = 0
            #print(length)
            for p in done.keys():
                if l in done[p] and p != key and p != length:
                    done[length].extend(done[p][:-1])
                    count += done[p][-1].get('count')
                    temp.append(p)
                    a += 1
            count -= a
            done[length] = list(np.unique(done[length]))
            done[length].append({'count' : count})
            
            for i in b:
                if l in done[i] and i != key:
                    l = i
            Z.append([key, l, small, done[key][-1]['count']+1])
            
            
            del(done[key])
            for i in temp:
                del(done[i])
            length += 1
            
            #print(Z)
            #print(done)
        else:
            Z.append([x, y, small, 2])
            done[length] = []
            done[length].append(x)
            done[length].append(y)
            done[length].append({'count' : 2})
            length += 1
            #print(done)
    else:
        a = []
        value = 0
        for i in done.keys():
            value += done[i][-1]['count']
            a.append(i)
        if len(a) == 1:
            a.append(x if x not in done[length-1] else y)
        a.append(small)
        a.append(value)
        Z.append(a)
        #print(done)
    #-----------------------------------------------------
    return x, y


def clustering(x, y):
    tmp = x if x < y else y
    p = y if x < y else x
    
    temp = np.reshape(np.zeros((len(dist)-1,len(dist)-1)), (len(dist)-1,len(dist)-1))
    temp = pd.DataFrame(dist, columns=list(range(len(dist)-1)), index=list(range(len(dist)-1)))
    temp.columns = [l for l in dist.columns if l != p]
    temp.index = [l for l in dist.index if l != p]
    
    for i in temp.index:
        for j in temp.columns:
            if i == tmp:
                temp[i][j] = min(dist[x][j],dist[y][j])
                temp[j][i] = temp[i][j]
            elif i < tmp:
                temp[i][j] = dist[i][j]
                temp[j][i] = temp[i][j]
            else:
                if j == tmp:
                    temp[i][j] = min(dist[i][x],dist[i][y])
                    temp[j][i] = temp[i][j]
                else:
                    temp[i][j] = dist[i][j]
                    temp[j][i] = temp[i][j]
    return temp


@jit(nopython=True)
def intersection(lst1, lst2):
    if len([value for value in lst1 if value in lst2]) > 0:
        return 1
    else:
        return 0


def update_clusters():
    global abc
    tempo = {}
    g = 0
    i = 0
    #print(abc)
    while(i < len(abc)):
        j = i+1
        while(j < len(abc)):
            if intersection(list(abc[i].keys()), list(abc[j].keys())):
                tempo = {**abc[i], **abc[j]}
                tempo = dict.fromkeys(tempo, colors[list(tempo.keys())[0]])
                abc.append(tempo)
                del abc[j]
                del abc[i]
                g=1
            j += 1
        if g:
            i = 0
            g = 0
        else:
            i += 1
            

@jit
def plot():
    plt.figure()
    for i in abc:
        for j in i.keys():
            plt.scatter(df['x'][j], df['y'][j], c=i[j], s=20)
    plt.show()

@jit
def compute_center(lst):
    sum = 0
    index = 0
    small = math.inf
    for i in lst:
        for j in lst:
            sum += euclidean(df['x'][i], df['y'][i], df['x'][j], df['y'][j])
        if sum < small:
            small = sum
            index = i
        sum = 0
    return index, small

@jit
def homogeneity():
    global abc
    homo = 0
    for i in range(len(abc)):
        index, small = compute_center(list(abc[i].keys()))
        for j in list(abc[i].keys()):
            homo += euclidean(df['x'][index], df['y'][index], df['x'][j], df['y'][j])
        homo /= len(abc[i].keys())
        print("Homogeneity Of Cluster %d is %.2f"%(i,homo))
        homo = 0
    #return homo

@jit
def separation():
    global abc
    sum = 0
    N = 1
    p = 0
    
    
    for i in range(len(abc)):
        for j in range(len(abc)):
            if i != j:
                sum += len(list(abc[i].keys())) + len(list(abc[j].keys()))
    N = sum            
    for i in range(len(abc)):
        for j in range(len(abc)):
            if i != j:
                index1, small1 = compute_center(list(abc[i].keys()))
                index2, small2 = compute_center(list(abc[j].keys()))
                p = len(list(abc[i].keys())) * len(list(abc[j].keys())) * euclidean(df['x'][index1], df['y'][index1], df['x'][index2], df['y'][index2])
                print("The Separation Between Cluster %d and %d is %.2f"%(i, j, p/N))
                p = 0
    #N = sum
    #return p/N


def ahc_single():
    global dist
    global intermediate_arrays
    global abc
    create_distance_matrix()
    #print("Dist Mat : ")
    #print(dist)
    round = 1
    plt.scatter(df['x'], df['y'], s=20)
    while len(dist) != 1:
        print("Round %d...."%(round))
        x, y = update_distance_matrix()
        print("i = %d j = %d"%(x, y))
        temp = {}
        temp[x] = {colors[x if x < y else y]}
        temp[y] = {colors[x if x < y else y]}
        abc.append(temp)
        #print("Initial Clusters : ")
        #print(abc)
        if round == 1:
            for j in range(len(df)):
                b = {}
                for i in abc:
                    if j in list(i.keys()):
                        global yes
                        yes += 1
                if yes == 0:
                    b = (dict({j: colors[j]}))
                    abc.append(b)
                else:
                    yes = 0
        update_clusters()
        #print("Final Clusters : ")
        print(abc)
        if round == 1:
            print("Initial Points Are : ")
        plot()
        homogeneity()
        #print("Homogeneity Of Cluster is %.2f"%(homogeneity()))
        if round+1 != len(df):
            separation()
            #print("Separation Of Clusters is %.2f\n\n"%(separation()))
        
        dist = clustering(x, y)
        #print("Dist Mat : ")
        #print(dist)
        intermediate_arrays.append(dist)
        round += 1


if __name__ == "__main__":
    start = time.time()
    ahc_single()
    end = time.time()
    print(end-start)
    Z = np.asarray(Z)
    #print(Z)


    labelList = range(0, len(df))

    plt.figure(figsize=(10, 7))  
    dendrogram(Z,  
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
    plt.show()  