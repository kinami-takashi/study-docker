#! /usr/bin/env python
#

from colorama import Fore, Back, Style  #colaboratoryで実行する場合、コメントアウト
import numpy
from scipy import signal

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ptick	#軸の目盛りを指数表記
import pylab	#グラフの指定範囲を塗りつぶす

import pandas as pd
import sympy as sym
import sympy
from statistics import mean, median,variance,stdev

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from chainer.datasets import split_dataset_random
from chainer import serializers

import array
import sys
import numpy as np
from numpy import nan as NA
from numpy.random import *

import math

from chainer.cuda import to_gpu
from chainer.cuda import to_cpu

import scipy.stats

from time import sleep

import csv
import pprint

#windowsからターミナルを使って、VMのlinux環境へリモートで作業していた描画の表示には、linuxのデスクトップが必要
#import matplotlib as mpl
#mpl.use('Agg')

#from sklearn.metrics import mean_squared_error	#二乗平均平方根誤差(RMSE)用


#ネットワーク
class Autoencoder(chainer.Chain):
    def __init__(self):
        super(Autoencoder, self).__init__(#encoder = L.Linear(16, 8),decoder = L.Linear(8, 16))	#元
                encoder = L.Linear(8, 6),decoder = L.Linear(6, 8))	#使用する特徴量8種類
				#encoder = L.Linear(6, 3),decoder = L.Linear(3, 6))	#卒論でのユニット数：入力層6、中間層3、出力層6

    def __call__(self, x, hidden=False):
        h = F.relu(self.encoder(x))
        if hidden:
            #print(h)
            return h
        else:
             #print(self.decoder(h))
             return self.decoder(h)


#標準化(複数あるデータの平均を0、分散が1になるように変換すること)
def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result



HF = 0
HF2 = 0
HF_test = 0
LF = 0
LF2 = 0
LF_test = 0
dss  = []
dsss = []
data = []
d = 0
b = 0
c = 199
bb = 0
cc = 150
N = 200		#分割数
n = 58175	#サンプル数
a = n/N
int(a)
#print(a)	#290.875
dt = 0.001
fs = 1/dt	#サンプリング周波数
sa = []
count = 0
count2 = 0
count_test = 0
dataset  = []
dataset1 = []
dataset2 = []
dataset3 = []
dataset4 = []
dataset5 = []
w = 0
y = 200


#評価用のデータ作成に使用
#csvファイルを読み込みいくつか処理を経てds_testにデータを格納
ds_test = pd.read_csv('/code/hyouka.csv')	#hyouka.csv:ストレス時,後半少し安静時,5007行1列（1行目のRRIラベル除く）
ds_test = ds_test['RRI']			#csvファイルからラベルRRI列を取得
#print("ds_test補間前--------------------------------------------------")
#print(ds_test)
ds_test = ds_test.interpolate(limit_direction='forward', limit_area='inside')	#線形補間
#print("ds_test補間後--------------------------------------------------")
#print(ds_test)
#print('hyouka.csv の次元数＝ ' + str(ds_test.ndim))			#1 (ndimで整数値intとして次元数取得)
#print('hyouka.csv のサイズ(行,列)＝ ' + str(ds_test.shape))	#(5007,)=5007×1（shapeで各次元のサイズ(行,列)取得)
#print(type(ds_test)) 											#<class 'pandas.core.series.Series'>


#評価用のデータ作成(aeの学習終了後,安静時とストレス時の心拍データの比較に使用,学習用データからランダムに抜粋)

for i in range(0,4):	#moto: range(0,2000):

	#print('i is ' + str(i))	#i is 0 〜 3

	ds_test2 = ds_test[w:y]	#補間後のhyouka.csv(ds_test)のw〜y行目を抽出
	#print('ds_test2--------------------------------------------------')
	#print(ds_test2)
	#print('分散を求める時に使用するds_test2 のサイズ(行,列)＝ '+ str(ds_test2.shape))
	#↪(i is 0→200行,i is 1→398行,i is 2→794(398+198×2)行,i is 3→1586(794+198×4)行,)
	s_test = sum(ds_test2)
	mean_test = s_test / N			#RRIの平均①つ目の心拍特徴量
	#print('RRIの平均 ＝ '+ str(mean_test))
	var_test = variance(ds_test2)	#RRIの分散②
	#print('RRIの分散 ＝ '+ str(var_test))
	dev_test = math.sqrt(var_test)	#RRIの標準偏差③
	#print('RRIの標準偏差(dev) ＝ '+ str(dev_test))
	std_test = stdev(ds_test2)		#RRIの標準偏差④
	#print('RRIの標準偏差(std) ＝ '+ str(std_test))	
	#rmse_test = np.sqrt(mean_squared_error(ds_test2[w], ds_test2[w+1]))	#隣接するRRIの二乗平均平方根誤差
	#print('RRIの二乗平均平方根誤差 = '+ str(rmse_test))


	for p in range(w,y):	#w = 0,y = 200 (i is 0)
		#print('p is ' + str(p))	#p is number = ds_test2
		#↪(i is 0→p is 0〜199, i is 1→p is 199〜596, i is 2→p is 596〜1389, i is 3→p is 1389〜2974)
		try:
			data.append(ds_test2[p])
		except KeyError:
			break
	y_test = np.array(data)					
	freq1, P1 = signal.welch(y_test, fs)	#welch法 : データ分割→離散フーリエ変換→平均（Rriからの各データ算出.pdfのp.4）		
	#print('freq1 is ' + str(freq1))
	

	#検証用データ作成(aeの学習ができているかの確認に使用)
	for q in range(0,41): 

		#print('q is ' + str(q))	#q is 0 〜 q is 40
		if  5 < freq1[q] < 15:
			HF_test = HF_test + P1[q]	#HF_test : PSDの高周波領域(0.15Hz - 0.4Hz)のパワー⑤
			#print('HF ＝ '+ str(HF_test))
		
		elif 15 < freq1[q] < 40:
			LF_test = LF_test + P1[q]	#LF_test :  PSDの低周波領域(0.04Hz - 0.15Hz)のパワー⑥
			#print('LF ＝ '+ str(LF_test))
			
	LFHF_test = LF_test/HF_test			#LFHF_test :  LFとHFの比⑦
	#print('LFHF ＝ '+ str(LFHF_test))


	y = y - 1	#1回目 y = 200 - 1 = 199
	#print('y is ' + str(y))
	#↪(i is 0→y is 199, i is 1→y is 596, i is 2→y is 1389, i is 3→y is 2974, i is 4→y is 6143,)

	for r in range(w,y):	#1回目 w = 0,y = 199

		#print('r is ' + str(r))	#r is ○ : ○ = ds_test2
		#↪(i is 0→r is 0〜198, i is 1→r is 199〜595, i is 2→r is 596〜1388)
		try:
			sa = ds_test[r] - ds_test[r+1]
		except KeyError:
			break
		if sa > 0.005:
			count_test = count_test + 1		#count_test : 一定時間内に隣接するRRIの差が50msecを超えるペアの個数⑧
			#print('count ＝ '+ str(count_test))
					
		w = w + 1	#1回目 w = 0 + 1 = 1
		#print('w is ' + str(w))		#w is ○ : ○ = ds_test2
		#↪(i is 0→w is 1,2〜199, i is 1→w is 200〜596, i is 2→w is 597〜1389,i is 3→w is 1390〜2974,)
		
		y = y + 2	#1回目 y = 199 + 2 = 201
		#print('y is ' + str(y))		#y is
		#↪(i is 0→y is 201,203〜597, i is 1→y is 598〜1390, i is 2→y is 1391〜2977,i is 3→y is 2976〜6144,)		

		if i == 0:
			test_st = [mean_test,var_test,dev_test,std_test,HF_test,LF_test,LFHF_test,count_test]
			#print(test_st)
			test_st = np.asarray([test_st],dtype='float32')	#asarray:numpy配列のコピーを作る
			#test_st = min_max(test_st)
			#print(test_st)

		elif i > 0:
			test_st2 = [mean_test,var_test,dev_test,std_test,HF_test,LF_test,LFHF_test,count_test]
			test_st2 = np.asarray([test_st2],dtype='float32')	#float32:32ビットの浮動小数点数
			test_st = np.append(test_st,test_st2,axis=0)	#append():リストの末尾に要素を追加,axis=0:列に沿った処理

#print('test_st のサイズ(行,列)＝ ' + str(test_st.shape))	#(2776, 8)
#print('test_st = ')
#print(test_st)

test_st = min_max(test_st)
#print('min_max(test_st) のサイズ(行,列)＝ ' + str(test_st.shape))	#(2776, 8)
#print(test_st)

#print('w ＝ ' + str(w))	#w = 2974
#print('y ＝ ' + str(y))	#y = 6144



w = 0
y = 200

for d in range(1):	#元 range(0,4):

	#print('d is ' + str(d))	#d is 0 〜 3

	for z in range(0,1935):	#元 range(0,7): → (0,1586):
	#hyouka21に合わせるとrange(0,1586):だがhyouka22は1196行の為、エラー発生

		#print('z is ' + str(z))	#z is 0 〜 6

		if d == 0:
			ds_test = pd.read_csv('/code/hyouka21.csv')	#テーマ実験？(30分間のRRI)
			ds_test = ds_test['RRI']	#csvファイルからラベルRRI列を取得
			ds_test = ds_test.interpolate(limit_direction='forward', limit_area='inside')	#線形補間
			#print('hyouka21.csv のサイズ＝ ' + str(ds_test.shape))	#(1786,) → (2135,)
			#print(ds_test)
			#print('hyouka21.csv の次元数＝ ' + str(ds_test.ndim))	#1
			#1次元のデータを書き込むとエラー発生
			#with open('/home/milab/デスクトップ/1_fitbit_stress/hyouka21_hokan.csv', 'w') as f:
			#	writer = csv.writer(f)		#csv.writer : csvファイルの書き込み（出力）
			#	writer.writerows(ds_test)		#writer.writerow(書き込む配列)


		#print('ds_test--------------------------------------------------')
		#print(ds_test)
		#print('w ＝ ' + str(w))	#w = w + 20
		#print('y ＝ ' + str(y))	#y = y + 20
		ds_test = ds_test[w:y]	#y - w = 200
		#print('ds_test のサイズ(行,列)＝ ' + str(ds_test.shape))	#(200,)	
		#print('ds_test = ds_test[w:y]-----------------------------------')
		#print(ds_test)
		s_test = sum(ds_test)
		mean_test = s_test / N
		#print('分散を求める時に使用するds_test のサイズ(行,列)＝ '+ str(ds_test.shape))
		var_test = variance(ds_test)
		dev_test = math.sqrt(var_test)
		std_test = stdev(ds_test)
		
		for p in range(w,y):

			#print('p is ' + str(p))
			try:
				data.append(ds_test[p])
			except KeyError:
				pass
		y_test = np.array(data)
		
		freq1, P1 = signal.welch(y_test, fs)
		#print('freq1 is ' + str(freq1))

		#検証用データ作成
		for q in range(0,41):

			#print('q is ' + str(q))
			if  5 < freq1[q] < 15:
				HF_test = HF_test + P1[q]
		
			elif 15 < freq1[q] < 40:
				LF_test = LF_test + P1[q]
		
		LFHF_test = LF_test/HF_test
		
		y = y - 1	#隣接するRRI(count_test)は199点のデータから求める
		#print('y = ' + str(y))
		#print('w = ' + str(w))
		#↪
		
		for r in range(w,y):	#1回目 w = 0,y = 199
			
			try:
				#print('r is ' + str(r))
				sa = ds_test[r] - ds_test[r+1]
				if sa > 0.005:
					count_test = count_test + 1
			except KeyError:
				pass
		
		w = w + 1	#w = w + 20
		#print('w = ' + str(w))
		#↪

		y = y + 2	#y = y + 21 
		#print('y = ' + str(y))
		#↪

		if z == 0:
			test_st = [mean_test,var_test,dev_test,std_test,HF_test,LF_test,LFHF_test,count_test]
			#print(test_st)
			test_st = np.asarray([test_st],dtype='float32')
			#test_st = min_max(test_st)
			#print('(if) test_st のサイズ＝ ' + str(test_st.shape))	#(1,8)
			#print(test_st)
		
		elif z > 0:		#(z == 1〜6)
			test_st2 = [mean_test,var_test,dev_test,std_test,HF_test,LF_test,LFHF_test,count_test]
			test_st2 = np.asarray([test_st2],dtype='float32')
			test_st = np.append(test_st,test_st2,axis=0)
			#print('(elif) test_st のサイズ＝ ' + str(test_st.shape))	#(z+1,8) → (1586,8)
			#print(test_st)
	

	#print('test_st のサイズ＝ ' + str(test_st.shape))	#(7,8) → (1586,8)
	#print('test_st =')
	#print(test_st)

	test_st = min_max(test_st)	#値変化
	#min_max : データの中における最大値と最小値を使って正規化する方法。この処理をすることでデータは最大値が1、最小値が0のデータとなる。
	#print('min_max(test_st) のサイズ＝ ' + str(test_st.shape))	#(7,8) → (1586,8)
	#print('test_st =')
	#print(test_st)


	if d == 0:
		test_kanada2_st = test_st	#test_st = hyouka21.csvの特徴量(7,8)
		#ds_test = pd.read_csv('kanada_st.csv')		#(998,1)
		#ds_test = ds_test['RR']
		#ds_test = ds_test.interpolate(limit_direction='forward', limit_area='inside')
		print('test_kanada2_st のサイズ＝ ' + str(test_kanada2_st.shape))	#(7,8) → (1586,8)
		#print('test_kanada2_st')
		#print(test_kanada2_st)		#test_kanada2_st = min_max(test__st)
		#with open('/home/ogawareiya/1sotuken/1_stress_estimation/1_data/test_kanada2_st.csv', 'w') as f:
		#	writer = csv.writer(f)		#csv.writer : csvファイルの書き込み（出力）
		#	writer.writerows(test_kanada2_st)		#writer.writerow(書き込む配列)

		w = 0
		y = 200


test_st = np.append(test_st,test_kanada2_st,axis=0)
print('test__st のサイズ＝ ' + str(test_st.shape))	#(14,8) → (3172,8) 3172=1586*2 

dataset = pd.read_csv('/code/dataset.csv')	#/ホーム/1_fitbit_stress内のdataset.csv
#print('dataset.csvを読み込んだdatasetのサイズ(行,列)＝' + str(dataset.shape))	#(13199,8)

dataset = np.asarray(dataset,dtype='float32')
#print('Numpy配列に変換したdatasetのサイズ(行,列)＝' + str(dataset.shape))		#(13199,8)

#'''

#安静時の特徴量dataset(13199,8)を評価用(testdata)と学習用(train)に分ける
testdata = dataset[0:2000]
train = dataset[2001:13199]		#元→[2001:13200]


#学習用データ作成
train = tuple_dataset.TupleDataset(train, train)
test = tuple_dataset.TupleDataset(testdata, testdata)
train_iter = chainer.iterators.SerialIterator(train, 64)
test_iter = chainer.iterators.SerialIterator(test, 16, False, False)


#モデル作成
model = L.Classifier(Autoencoder(), lossfun=F.mean_squared_error)
model.compute_accuracy = False
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)


#学習（バッチサイズ32）
N_EPOCH = 500	#エポック数（1つの訓練データを何回繰り返し学習するか）

updater = training.StandardUpdater(train_iter, optimizer, device=0)
trainer = training.Trainer(updater, (N_EPOCH, 'epoch'), out="result")

trainer.extend(extensions.LogReport())
trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}'))
trainer.extend(extensions.Evaluator(test_iter, model, device=0))
trainer.extend(extensions.PrintReport( ['epoch', 'main/loss',  'validation/main/loss', 'elapsed_time']),trigger=(10, 'epoch'))
trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch',marker = 'None', file_name='loss.eps'))
trainer.extend(extensions.ProgressBar())

print('学習開始---------------------------------------------------')
trainer.run()
print('学習終了---------------------------------------------------')

sleep(1)

#学習済みモデルの保存、読み込み
serializers.save_npz("mymodel.npz", model)
serializers.load_npz('/code/mymodel.npz', model)


#検証
gosa = []
gosa2 = []
n = 0
x = 0
a = []


#安静時とストレス時の心拍特徴量を学習済aeに入力
#aeに入力する前の心拍特徴量とaeにより復元した心拍特徴量の差(rmse)を計算
#for s in range (): のかたまりがそれぞれ評価実験や安静時

print('テーマ実験の復元誤差算出')
total_gosa3 = []
for s in range (0,1935):	#元 range (0,7): →  (0,1586)

		#print('s is ' + str(s))
		
		#dataa = ae入力前の心拍特徴量（実際の値）
		dataa = test_kanada2_st[s]		#test_kanada2_st=(1586,8) , テーマのrri(hyouka21.csv)から特徴量算出
		#print('dataaのサイズ＝' + str(dataa.shape))	#(8,) , test_kanada2_stの1列目の8つの値(s is 1)
		#print('dataa = test_kanada2_st[s] (before)')
		#print(dataa)
	
		#datab = aeにより復元した心拍特徴量（正解の値）
		model.to_gpu(0)		#GPU上で計算させるためにモデルをGPUに送る(GPU対応にする)
		datab = to_gpu(dataa[None, ...])
		datab = model.predictor(datab)	#model.predictor : 予測後の出力ノードの配列を生成(https://ai-trend.jp/programming/python/chainer-neural-network/)
		datab = datab.array				#一次元配列作成
		datab = to_cpu(datab)			#cpuで処理
		#print('databのサイズ＝' + str(datab.shape))	#(1,8)
		#print('datab (after)')
		#print(datab)
     
		#gosa3 = rmse , gosa3(1,8)の8つの値の平均が復元誤差
		gosa3 = dataa - datab			#ae入力前の心拍特徴量(dataa)とaeにより復元した心拍特徴量(datab)の誤差を求める
		gosa3 = gosa3 * gosa3			#誤差を二乗
		#print("heikinmae gosa3")				
		#print(gosa3)					#(1,8)
		gosa3 = np.mean(gosa3)			#平均を取る
		gosa3 = math.sqrt(gosa3)		#平方根を取る
		#print('gosa3のサイズ＝' + str(gosa3.size()))	#(1,8)×1587=(1587,8) , error
		#print('heikinngo gosa3')
		#print(gosa3)					#(1,1)
		total_gosa3.append(gosa3)
		#with open('/home/milab/デスクトップ/1_fitbit_stress/gosa3_2.csv', 'a') as f:
			#writer = csv.writer(f)		#csv.writer : csvファイルの書き込み（出力）
			#writer.writerow(str(gosa3))		#writer.writerow(書き込む配列


#テーマ実験のrri(theme)と復元誤差(total_gosa3)の2種類の折れ線グラフを作成する
theme = pd.read_csv('/code/hyouka22.csv')	#テーマ実験(30分間のRRI)
theme = theme['RRI']				#csvファイルからラベルRRI列を取得
theme = theme.interpolate(limit_direction='forward', limit_area='inside')	#線形補間
print('hyouka22.csv のサイズ＝ ' + str(theme.shape))	#(2135,)
#print('hyouka21_same_nisikata.csv の次元数＝ ' + str(theme.ndim))	#1
#print(theme)


print("配列の大きさ")
print(len(theme))			#2135
print(len(total_gosa3))		#1900（1900個の復元誤差を使用して、アブストラクト図3の折れ線グラフ作成）


#２つのグラフを左側の第一軸(ax1)と右側の第二軸(ax2)を作って,同じx軸に関連付ける
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

plt.figure						#グラフの描画先の準備
ax1.plot(theme, color = "blue", label = "rri")		#グラフを描く(rri)
ax2.plot(total_gosa3, color = "red", label = "reconstraction_error")	#グラフを描く(復元誤差) 

ax1.set_xticks([0, 300, 600, 900, 1200, 1500, 1800, 2100])	#x軸の目盛設定
ax1.set_yticks([0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8])			#y軸の目盛設定(左側)
ax2.set_yticks([0.00001, 0.000015, 0.00002, 0.000025, 0.00003, 0.000035, 0.00004, 0.000045, 0.00005])			#y軸の目盛設定(右側)

#time = [0, 300, 600, 900, 1200, 1500, 1800]	#x軸の値を指定
#reconstraction_error = []	#y軸の値を指定(グラフの右側)
#rri = []	#y軸の値を指定(グラフの左側)

ax2.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))	#10のべき乗は"ScalorFormatter"で"useMathText=1"
ax2.yaxis.offsetText.set_fontsize(10)						#指数文字は"offset"
ax2.ticklabel_format(style='sci',axis='y',scilimits=(0,0))	#指数表記

ax1.set_xlabel("time(s)")				#x軸のラベル
ax1.set_ylabel("RRI(s)")				#y軸のラベル(左側)
ax2.set_ylabel("reconstraction_error")	#y軸のラベル(右側)

#凡例を表示（upper left : 凡例の位置を左上に指定、ax2をax1のやや下に持っていく）
ax1.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0.5, fontsize=10)
ax2.legend(bbox_to_anchor=(0, 0.9), loc='upper left', borderaxespad=0.5, fontsize=10)

ax1.axvspan(0, 300, facecolor='m', alpha=0.3)	#axvspan(開始, 終了, facecolor=[塗りつぶす色], alpha=[透過度])
ax1.axvspan(600, 900, facecolor='m', alpha=0.3)
ax1.axvspan(1200, 1500, facecolor='m', alpha=0.3)
ax1.axvspan(1800, 2135, facecolor='gray', alpha=0.3)

plt.savefig("/code/theme_hukugen_gosa_and_rri.png")	#保存
plt.show()										#グラフを表示