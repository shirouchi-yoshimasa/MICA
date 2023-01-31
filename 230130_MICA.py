import streamlit as st
import datetime
from io import StringIO
import io
import itertools
from PIL import Image
from scipy import stats
import pandas as pd
import scipy
import time
from module.module import Convolution
from module.module import fft_plot
from module.module import date_time_trans
from module.module import date_time_trans2
from module.module import trigger_cut
from module.module import unnamed_drop
from module.module import connect_s3_skkeng
from module.module import fn_transport
from module.module import read_c_data
from module.module import pre_data_set
from module.module import mahalnobis_distance
from module.module import md_score
from module.module import MD_plot
from module.module import MD_score_plot

st.title('土工MAS振動解析')

maslist = ['1-1','1-2','2-1','2-2']

with st.form("my_form", clear_on_submit=False):
    MAS_name = st.selectbox(label='MAS機種指定してください', options=[f'RFD{i}' for i in maslist])
    sxmin = st.selectbox(label='グラフ開始時期を選んでください', options=['2022/6/10','2022/9/10','2022/12/10'])
    trigger = st.selectbox(label='トリガー値を入力してください(初期設定0.1)', options=[0.1,0.2,0.3,0.4,0.5])
    graph = st.selectbox(label='グラフ種類指定してください', options=['MD_SCORE','MAX'])
    submitted = st.form_submit_button("グラフ表示")

if submitted:
    with st.spinner("処理中です..."):
        time.sleep(1)
    if MAS_name == "RFD1-1":
        c_name = "c_1"
    elif MAS_name == "RFD1-2":
        c_name = "c_2"
    elif MAS_name == "RFD2-1":
        c_name = "c_3"
    else:
        c_name = "c_4"

    dt_now = datetime.datetime.now()
    now = dt_now.strftime('%Y%m%d')[2::]
    sxmax = dt_now.strftime('%Y/%m/%d')
    b_p = 1#MD閾値
    sb_p = 20#MDscore閾値
    #周波数帯1000Hz毎に区間平均で畳み込み
    n=10
    range_freq = [8192 // n * (i+1) for i in range(n-1)]
    range_freq.append(8192)
    range_freq.insert(0,0)

    s3,bucket = connect_s3_skkeng()
    df_fn = fn_transport(s3)
    fn = df_fn.values.tolist()
    fn = list(itertools.chain.from_iterable(fn))
    fn = [i[:-1] for i in fn]
    files_all = [obj_summary.key for obj_summary in bucket.objects.all()]
    files_fft = [i for i in files_all if '.fft' in i]
    files_fft2 = set(files_fft) - set(fn)
    files_fft2 = list(files_fft2)
    with open('filename.txt', mode='w') as f:
        f.writelines('\n'.join(files_fft))
    c = read_c_data(c_name,s3)
    c_1 = pre_data_set(files_fft2,c_name,c,s3,trigger,range_freq)
    c_1 = c_1.dropna(axis=1)# axis = 1 で，欠損値のある列を削除
    c_1 = unnamed_drop(c_1)
    c_1.to_csv(f"{c_name}.csv")
    dt2 = c_1.columns.values.tolist()#データ取得時間一覧リスト化
    dte2 = date_time_trans2(dt2)
    m_1 = mahalnobis_distance(c_1)
    df_m1 = pd.DataFrame(list(zip(dte2,m_1)), columns = ['date','MD'])
    #b_p = 1#閾値
    df_m1_b = df_m1[df_m1['MD'] > b_p]
    #正規化 
    mds_1 = scipy.stats.zscore(m_1)
    ms_1 = md_score(mds_1)
    df_ms1 = pd.DataFrame(list(zip(dte2,ms_1)), columns = ['date','MD'])
    #sb_p = 50#閾値
    df_ms1_b = df_ms1[df_ms1['MD'] < sb_p]
    
    if graph == 'MAX':
        y_max = max(m_1)
        MD_plot(df_m1,df_m1_b,y_max,MAS_name,sxmin,sxmax)
        image = Image.open(f'{MAS_name}MAXE.png')
        st.image(image, caption=f'{MAS_name}MAX',use_column_width=True)
    else:
        y_max = max(ms_1)
        MD_score_plot(df_ms1,df_ms1_b,MAS_name,sxmin,sxmax)
        image = Image.open(f'{MAS_name}MD_SCORE.png')
        st.image(image, caption=f'{MAS_name}MD_SCORE',use_column_width=True)

    s3.Bucket('skkeng').upload_file(Filename='filename.txt', Key='filename.txt')
    s3.Bucket('skkeng').upload_file(Filename=f"{c_name}.csv", Key=f"{c_name}.csv")