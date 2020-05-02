# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:41:36 2019

@author: SAMSUNG
"""
import pandas as pd
import cv2 as cv
import re
import glob
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def to_df(path):
    df=pd.DataFrame(columns=['person', 'face'])
    i=0
    for pic in path:
        name=re.findall('\\\\(.*)\\\\',pic)[0]
        face=cv.imread(pic,cv.IMREAD_GRAYSCALE).reshape(-1)
        df.loc[i]=dict({'person':name, 'face':face})
        i+=1
    return df

def prepare(gender):
    for x in range(len(gender)):
        hist_img = cv.equalizeHist(gender.iloc[x, 1])
        hist_img = np.reshape(hist_img, -1)
        gender.iloc[x, 1] = hist_img

def pca(gender):
    n=15
    model = PCA(n).fit(np.array(gender.loc[:, 'face'].to_list()))
#    model= PCA(15).fit(face_vecs)
    gender_de = pd.concat([gender.loc[:,'person'], pd.DataFrame(model.transform(np.array(gender.loc[:, 'face'].to_list())),columns = range(n))],axis = 1)
    return gender_de, model

def nmf(gender):
    n=15
    model = NMF(n).fit(np.array(gender.loc[:, 'face'].to_list()))
#    model= PCA(15).fit(face_vecs)
    gender_de = pd.concat([gender.loc[:,'person'], pd.DataFrame(model.transform(np.array(gender.loc[:, 'face'].to_list())),columns = range(n))],axis = 1)
    return gender_de, model

def task(gender, n=1):
#    n = 1 时准确率最高
    x_train, x_test, y_train, y_test = train_test_split(gender.iloc[:,1:], gender['person'], test_size = 0.9)
    knn = KNeighborsClassifier(n).fit(x_train, y_train)
#    pre_train = knn.predict(x_train)
#    pre_test = knn.predict(x_test)
    score_train = knn.score(x_train, y_train)
    score_test = knn.score(x_test, y_test)
    print('Train score: {}\nTest score: {}'.format(score_train, score_test))




#读取
male_path=glob.glob('faces94/male/*/*.jpg')
female_path=glob.glob('faces94/female/*/*.jpg')
male_raw = to_df(male_path)
female_raw = to_df(female_path)

#未预处理的降维
male_de, pca_male=pca(male_raw)
female_de, pca_female=pca(female_raw)
'''
male_de, nmf_male=nmf(male_raw)
female_de, nmf_female=nmf(female_raw)
'''

#预处理
prepare(male_raw)
prepare(female_raw)

#预处理后的降维
male_dep, pca_malep=pca(male_raw)
female_dep, pca_femalep=pca(female_raw)

#k值的选取
#dm = {}
#df = {}
#for x in range(1,10):
#    dm[x] = task(male_de, x)
#    df[x] = task(female_de, x)
#plt.figure(1)
#plt.plot(dm.keys(),np.array(list(dm.values()))[:,0])
#plt.show()
#plt.plot(dm.keys(),np.array(list(dm.values()))[:,1])
#plt.show()
#plt.figure(2)
#plt.plot(df.keys(),np.array(list(df.values()))[:,0])
#plt.show()
#plt.plot(df.keys(),np.array(list(df.values()))[:,1])
#plt.show()

#输出识别结果
print('Male:')
task(male_de)
print('\nFemale:')
task(female_de)

#特征脸
'''
eigen1=np.zeros(15)
face1=np.around(pca_male.inverse_transform(eigen1)).reshape(200,180)
plt.imshow(face1, cmap='gray')
plt.show()

eigen1=np.ones(15)
face1=np.around(nmf_male.inverse_transform(eigen1)).reshape(200,180)
plt.imshow(face1, cmap='gray')
plt.show()
'''
