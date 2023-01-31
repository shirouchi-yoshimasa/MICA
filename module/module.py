import streamlit as st
import math
import pandas as pd
import altair as alt
import itertools
#import chardet
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import numpy.random as random
import seaborn as sns
import scipy as sc
import scipy as sp
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from scipy import linalg
from scipy import spatial
#from scipy import stats
import scipy.spatial.distance
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager
#%matplotlib inline
import os
import glob
import datetime
from scipy import signal
import math
import boto3
from io import StringIO
import io
import time
from PIL import Image

def Convolution(df,dte,range_freq,range_Hz):
    L1 =[]
    for k in range(len(df.columns)):
        L = []
        for i in range(len(range_freq)-1):
            a = 0
            for j in range(range_freq[i+1] - range_freq[i]):
                a += df[dte[k]].iloc[j+range_freq[i]]
            L.append(a)
        L1.append(L)
    df3 = pd.DataFrame(L1)
    df3 = df3.T
    df3.index = [str(i*range_Hz)+'-'+str((i+1)*range_Hz)+'Hz' for i in range(len(range_freq)-1)]
    df3.columns = dte
    return df3

def fft_plot(df,MAS,n):
    n = min(n,len(df.columns))
    ncol = min(n,24)# グラフの横の数を求める
    nrow = int(np.ceil(n/ncol))# グラフの縦の数を求める
    df2 = df.iloc[:,-n:]
    fig = plt.figure(figsize=(4*n,0.4*n))# グラフの描画処理
    for i, (name, d) in enumerate(df2.iteritems()):# ループ処理
        ax = plt.subplot2grid((nrow, ncol), (i//ncol, i%ncol))
        d.plot(ax=ax) # ax.plot(d) でも可
        plt.ylim([0,0.5])
        ax.set_title(MAS+'_'+name)

def date_time_trans(dt):
    a = [dt[i][6:18] for i in range(len(dt))]
    dte = [datetime.datetime.strptime(a[i], '%Y%m%d%H%M') for i in range(len(a))]
    #dte.insert(0, 'fft_freq')
    return dte

def date_time_trans2(dt):
    a =[]
    for i in dt:
        if type(i) == str:
            a.append(datetime.datetime.strptime(i[0:19], "%Y-%m-%d %H:%M:%S"))
        else:
            b = i.strftime("%Y-%m-%d %H:%M:%S")
            a.append(datetime.datetime.strptime(b, "%Y-%m-%d %H:%M:%S"))
    return a

#トリガ以下削除
def trigger_cut(df,dte,trigger):
    dte_r = []
    for i in dte:
        a = list(df[i])
        if type(max(a)) != float:
            df = df.drop(i, axis=1)
        elif max(a) < trigger:
            df = df.drop(i, axis=1)
        else:
            dte_r.append(i)
    return df,dte_r

def unnamed_drop(df):
    a = df.columns.values.tolist()
    b = []
    for i in a:
        if 'Un' in str(i):
            b.append(i)
    df2 = df.drop(b,axis=1)
    return df2

def mahalnobis_distance(df):
    ROW = len(df)
    COLUMN = len(df.columns)
    row = []
    column = []
    ave = [0.0 for i in range(ROW)]
    vcm = np.zeros((COLUMN, ROW, ROW))
    diff = np.zeros((1, ROW))
    mahal_b = np.zeros(COLUMN)
    tmp = np.zeros(ROW)
    for i in range(ROW):# rowに要素をリストの形式で連結
        row.append(list(df.iloc[i]))
    for i in range(COLUMN):# 列を連結
        column.append(list(df.iloc[:, i]))
    for i in range(ROW):# 平均値の計算
        ave[i] = np.average(row[i][::len(row[i])])
    column = np.array([column])# Numpyのメソッドを使うので，array()でリストを変換した．
    ave = np.array(ave)
    for i in range(COLUMN):# 分散共分散行列を求める# np.swapaxes()で軸を変換することができる．
        diff = (column[0][i] - ave)
        diff = np.array([diff])
        vcm[i] = (diff * np.swapaxes(diff, 0, 1)) / COLUMN
    for i in range(COLUMN):# mahalnobis distanceを求める# 一般逆行列を生成し，計算の都合上転値をかける
        vcm[i] = sc.linalg.pinv(vcm[i])
        vcm[i] = vcm[i].transpose()
        vcm[i] = np.identity(ROW)
        diff = (column[0][i] - ave)# 差分ベクトルの生成
        for j in range(ROW):
            tmp[j] = np.dot(diff, vcm[i][j])
        mahal_b[i] = np.dot(tmp, diff)
    mahal = np.sqrt(mahal_b)
    return mahal

def md_score(md):
    ms = []
    for i in md:
        if math.isnan(i):
            ms.append(0)
        else:
            a = 1.0 / (1.0 + np.exp(-i))
            a = (1-a)* 100
            ms.append(a)
    return ms

def MD_plot(df,dfb,y_max,MAS,sxmin,sxmax):
    plt.figure(figsize=(20, 3), dpi=100)
    xmin = datetime.datetime.strptime(sxmin, '%Y/%m/%d')
    xmax = datetime.datetime.strptime(sxmax, '%Y/%m/%d')
    plt.xlim([xmin,xmax])
    plt.gcf().autofmt_xdate()
    plt.ylim([0,y_max*1.2])
    plt.ylabel('MD')
    plt.scatter(df['date'], df['MD'], color='black', label='All')
    plt.scatter(x=dfb['date'], y=dfb['MD'], color='red', label='Bad')
    plt.title(MAS+'_MD')
    #plt.savefig(now+'_'+MAS+'MD')
    #files.download(now+'_'+MAS+'.png')

def MD_score_plot(df,dfb,MAS,sxmin,sxmax):
    plt.figure(figsize=(20, 3), dpi=100)
    xmin = datetime.datetime.strptime(sxmin, '%Y/%m/%d')
    xmax = datetime.datetime.strptime(sxmax, '%Y/%m/%d')
    plt.xlim([xmin,xmax])
    plt.gcf().autofmt_xdate()
    plt.ylim([0,100])
    plt.ylabel('MD SCORE')
    plt.scatter(df['date'], df['MD'], color='black', label='All')
    plt.scatter(x=dfb['date'], y=dfb['MD'], color='red', label='Bad')
    plt.title(MAS+'_MD SCORE')
    plt.savefig(MAS+'MD_SCORE')
    #files.download(now+'_'+MAS+'MD_SCORE'+'.png')

def connect_s3_skkeng():
    s3 = boto3.resource('s3',
                           region_name="ap-northeast-1",
                           aws_access_key_id='AKIAV27ZCYO3NIGWWEEZ',
                           aws_secret_access_key='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'
                       )
    return s3,s3.Bucket('skkeng')

def fn_transport(s3):
    src_obj = s3.Object('skkeng','filename.txt')
    body_in = src_obj.get()['Body'].read().decode("utf-8")
    buffer_in = io.StringIO(body_in)
    df_fn = pd.read_csv(buffer_in,header = None, index_col=0,lineterminator='\n')
    df_fn.reset_index(inplace= True)
    return df_fn

def read_c_data(c,s3):
    src_obj = s3.Object('skkeng',f'{c}.csv')
    body_in = src_obj.get()['Body'].read().decode("utf-8")
    buffer_in = io.StringIO(body_in)
    return pd.read_csv(buffer_in,header = 0, index_col=0,lineterminator='\n')

def pre_data_set(files,c_name,c,s3,trigger,range_freq):
    if len(files) > 0:
        src_obj = s3.Object('skkeng',files[0])
        body_in = src_obj.get()['Body'].read().decode("utf-8")
        buffer_in = io.StringIO(body_in)
        df_out = pd.read_csv(buffer_in,header = None, index_col=0,lineterminator='\n')
        df_out.columns = [files[0][9::]]
        for i in range(len(files)-1):
            src_obj = s3.Object('skkeng',files[i+1])
            body_in = src_obj.get()['Body'].read().decode("utf-8")
            buffer_in = io.StringIO(body_in)
            if len(buffer_in.getvalue()) > 100000:
                df_in = pd.read_csv(buffer_in,header = None, index_col=0,lineterminator='\n')
                df_in.columns = [files[i+1][9::]]
                df_out = pd.concat([df_out, df_in], axis=1)
            else:
                continue
        df = df_out
        if c_name == 'c_1':
            df_1 = df.filter(regex='01001_')
            df_1 = df_1.dropna(axis=1)# axis = 1 で，欠損値のある列を削除
            dt_1 = df_1.columns.values.tolist()#データ取得時間一覧リスト化
            dte_1 = date_time_trans(dt_1)
            df_1.columns = dte_1[0::]
            df_1,dte_1 = trigger_cut(df_1,dte_1,trigger)
            n = 1000
            c_1 = Convolution(df_1,dte_1,range_freq,n)
            c_1_c = pd.concat([c_1,c], axis=1)
            return c_1_c
        elif c_name == 'c_2':
            df_1 = df.filter(regex='01002_')
            df_1 = df_1.dropna(axis=1)# axis = 1 で，欠損値のある列を削除
            dt_1 = df_1.columns.values.tolist()#データ取得時間一覧リスト化
            dte_1 = date_time_trans(dt_1)
            df_1.columns = dte_1[0::]
            df_1,dte_1 = trigger_cut(df_1,dte_1,trigger)
            n = 1000
            c_1 = Convolution(df_1,dte_1,range_freq,n)
            c_1_c = pd.concat([c_1,c], axis=1)
            return c_1_c
        elif c_name == 'c_3':
            df_1 = df.filter(regex='01003_')
            df_1 = df_1.dropna(axis=1)# axis = 1 で，欠損値のある列を削除
            dt_1 = df_1.columns.values.tolist()#データ取得時間一覧リスト化
            dte_1 = date_time_trans(dt_1)
            df_1.columns = dte_1[0::]
            df_1,dte_1 = trigger_cut(df_1,dte_1,trigger)
            n = 1000
            c_1 = Convolution(df_1,dte_1,range_freq,n)
            c_1_c = pd.concat([c_1,c], axis=1)
            return c_1_c
        else:
            df_1 = df.filter(regex='01004_')
            df_1 = df_1.dropna(axis=1)# axis = 1 で，欠損値のある列を削除
            dt_1 = df_1.columns.values.tolist()#データ取得時間一覧リスト化
            dte_1 = date_time_trans(dt_1)
            df_1.columns = dte_1[0::]
            df_1,dte_1 = trigger_cut(df_1,dte_1,trigger)
            n = 1000
            c_1 = Convolution(df_1,dte_1,range_freq,n)
            c_1_c = pd.concat([c_1,c], axis=1)
            return c_1_c
    else:
        return c

def mahalnobis_distance(df):
    ROW = len(df)
    COLUMN = len(df.columns)
    row = []
    column = []
    ave = [0.0 for i in range(ROW)]
    vcm = np.zeros((COLUMN, ROW, ROW))
    diff = np.zeros((1, ROW))
    mahal_b = np.zeros(COLUMN)
    tmp = np.zeros(ROW)
    for i in range(ROW):# rowに要素をリストの形式で連結
        row.append(list(df.iloc[i]))
    for i in range(COLUMN):# 列を連結
        column.append(list(df.iloc[:, i]))
    for i in range(ROW):# 平均値の計算
        ave[i] = np.average(row[i][::len(row[i])])
    column = np.array([column])# Numpyのメソッドを使うので，array()でリストを変換した．
    ave = np.array(ave)
    for i in range(COLUMN):# 分散共分散行列を求める# np.swapaxes()で軸を変換することができる．
        diff = (column[0][i] - ave)
        diff = np.array([diff])
        vcm[i] = (diff * np.swapaxes(diff, 0, 1)) / COLUMN
    for i in range(COLUMN):# mahalnobis distanceを求める# 一般逆行列を生成し，計算の都合上転値をかける
        vcm[i] = sc.linalg.pinv(vcm[i])
        vcm[i] = vcm[i].transpose()
        vcm[i] = np.identity(ROW)
        diff = (column[0][i] - ave)# 差分ベクトルの生成
        for j in range(ROW):
            tmp[j] = np.dot(diff, vcm[i][j])
        mahal_b[i] = np.dot(tmp, diff)
    mahal = np.sqrt(mahal_b)
    return mahal

def md_score(md):
    ms = []
    for i in md:
        if math.isnan(i):
            ms.append(0)
        else:
            a = 1.0 / (1.0 + np.exp(-i))
            a = (1-a)* 100
            ms.append(a)
    return ms

def MD_plot(df,dfb,y_max,MAS,sxmin,sxmax):
    plt.figure(figsize=(10, 2), dpi=100)
    xmin = datetime.datetime.strptime(sxmin, '%Y/%m/%d')
    xmax = datetime.datetime.strptime(sxmax, '%Y/%m/%d')
    plt.xlim([xmin,xmax])
    plt.gcf().autofmt_xdate()
    plt.ylim([0,y_max*1.2])
    plt.ylabel('MD')
    plt.scatter(df['date'], df['MD'], color='black', label='All')
    plt.scatter(x=dfb['date'], y=dfb['MD'], color='red', label='Bad')
    plt.title(MAS+'_MD')
    #plt.savefig(now+'_'+MAS+'MD')
    #files.download(now+'_'+MAS+'.png')

def MD_score_plot(df,dfb,MAS,sxmin,sxmax):
    plt.figure(figsize=(10, 2), dpi=100)
    xmin = datetime.datetime.strptime(sxmin, '%Y/%m/%d')
    xmax = datetime.datetime.strptime(sxmax, '%Y/%m/%d')
    plt.xlim([xmin,xmax])
    plt.gcf().autofmt_xdate()
    plt.ylim([0,100])
    plt.ylabel('MD SCORE')
    plt.scatter(df['date'], df['MD'], color='black', label='All')
    plt.scatter(x=dfb['date'], y=dfb['MD'], color='red', label='Bad')
    plt.title(MAS+'_MD SCORE')
    plt.savefig(MAS+'MD_SCORE')
    #files.download(now+'_'+MAS+'MD_SCORE'+'.png')













