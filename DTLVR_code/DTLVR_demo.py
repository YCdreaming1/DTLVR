import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from DiPLS import *
from di_pls import *
import dipals as ml1
from PLS import *
import DTLVR as ml
import seaborn as sns

def augmented_matrix(data, s, N):
    """
    Construct Augmented Matrix for feature data Z_s
    :param data: Original data
    :param s: int, lagged time
    :param N: int, Number of augmented samples
    :return:
    Z_s: numpy array, Augmented data matrices.
    """
    X=[]
    for i in range(s):
        X.append(data[i:i+N+1, :])
    Z_s = X[s-1]
    for i in range(s-2, -1, -1):
        Z_s = np.hstack((Z_s, X[i]))
    return Z_s


"加载多工况数值例子"

'工况1'
xk_mode1 = np.load(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\xk_mode1.npy')
yk_mode1 = np.load(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\yk_mode1.npy')

'工况2'
xk_mode2 = np.load(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\xk_mode2.npy')
yk_mode2 = np.load(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\yk_mode2.npy')

'工况3'
xk_mode3 = np.load(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\xk_mode2.npy')
yk_mode3 = np.load(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\yk_mode2.npy')

# plt.plot(np.vstack([yk_mode1,yk_mode3,yk_mode2]))

"划分训练集和测试集"
'工况1'
xk_mode1_train = xk_mode1[:400,:]
xk_mode1_test = xk_mode1[400:,:]
yk_mode1_train = yk_mode1[:400,:]
yk_mode1_test = yk_mode1[400:,:]

'工况2'
xk_mode2_train = xk_mode2[:400,:]
xk_mode2_test = xk_mode2[400:,:]
yk_mode2_train = yk_mode2[:400,:]
yk_mode2_test = yk_mode2[400:,:]

'工况3'
xk_mode3_train = xk_mode3[:400,:]
xk_mode3_test = xk_mode3[400:,:]
yk_mode3_train = yk_mode3[:400,:]
yk_mode3_test = yk_mode3[400:,:]

"数据标准化处理"
'工况1'
scaler1_x = MinMaxScaler()
xk_mode1_train_scale = scaler1_x.fit_transform(xk_mode1_train)
xk_mode1_test_scale = scaler1_x.transform(xk_mode1_test)
# xk_mode1_train_scale = xk_mode1_train
# xk_mode1_test_scale = xk_mode1_test
#
scaler1_y = MinMaxScaler()
yk_mode1_train_scale = scaler1_y.fit_transform(yk_mode1_train)
yk_mode1_test_scale = scaler1_y.transform(yk_mode1_test)
# yk_mode1_train_scale = yk_mode1_train
# yk_mode1_test_scale = yk_mode1_test
#
# '工况2'
scaler2_x = MinMaxScaler()
xk_mode2_train_scale = scaler2_x.fit_transform(xk_mode2_train)
xk_mode2_test_scale = scaler2_x.transform(xk_mode2_test)
# xk_mode2_train_scale = xk_mode2_train
# xk_mode2_test_scale = xk_mode2_test
#
scaler2_y = MinMaxScaler()
yk_mode2_train_scale = scaler2_y.fit_transform(yk_mode2_train)
yk_mode2_test_scale = scaler2_y.transform(yk_mode2_test)
# yk_mode2_train_scale = yk_mode2_train
# yk_mode2_test_scale = yk_mode2_test
#
# '工况3'
scaler3_x = MinMaxScaler()
xk_mode3_train_scale = scaler3_x.fit_transform(xk_mode3_train)
xk_mode3_test_scale = scaler3_x.transform(xk_mode3_test)
# xk_mode3_train_scale = xk_mode3_train
# xk_mode3_test_scale = xk_mode3_test
#
#
scaler3_y = MinMaxScaler()
yk_mode3_train_scale = scaler3_y.fit_transform(yk_mode3_train)
yk_mode3_test_scale = scaler3_y.transform(yk_mode3_test)
# yk_mode3_train_scale = yk_mode3_train
# yk_mode3_test_scale = yk_mode3_test
"===============================Model Training==========================================="

"--------------------------PLS model-----------------------------------------------------"
#潜变量数
l=3
#延迟窗口大小
s=2

# pls modeling
# pls = PLSRegression(n_components=l)
# pls.fit(xk_mode1_train_scale, yk_mode1_train_scale)
# pred_y1 = pls.predict(xk_mode3_test_scale)

pls1 = PLS(l=l)
params1 = pls1.train(xk_mode1_train_scale, yk_mode1_train_scale)
W = params1['W']
pred_y1 =pls1.predict(params1, xk_mode3_test_scale)
RMSE1=np.sqrt(mean_squared_error(yk_mode3_test_scale[s:],pred_y1[s:]))  # RMSE
print('PLS_RMSE:',RMSE1)

R1=r2_score(yk_mode3_test_scale[s:],pred_y1[s:])
print('PLS_R2:',R1)

"--------------------------DPLS model----------------------------------------------------"
#构建延迟矩阵
xk_mode1_train_scale_aug = augmented_matrix(xk_mode1_train_scale,s+1,xk_mode1_train_scale.shape[0]-s-1)
yk_mode1_train_scale_aug = yk_mode1_train_scale[s:yk_mode1_train_scale.shape[0]]
# print(yk_mode1_train_scale_aug.shape)

xk_mode3_test_scale_aug = augmented_matrix(xk_mode3_test_scale,s+1,xk_mode3_test_scale.shape[0]-s-1)
yk_mode3_test_scale_aug = yk_mode3_test_scale[s:yk_mode3_test_scale.shape[0]]

pls2 = PLS(l=l)
params2 = pls2.train(xk_mode1_train_scale_aug, yk_mode1_train_scale_aug)
pred_y2 =pls2.predict(params2, xk_mode3_test_scale_aug)
W_DPLS = params2['W']
RMSE2=np.sqrt(mean_squared_error(yk_mode3_test_scale[s:],pred_y2))  # RMSE
print('DPLS_RMSE:',RMSE2)

R2=r2_score(yk_mode3_test[s:],pred_y2)
print('DPLS_R2:',R2)




"--------------------------DiPLS model---------------------------------------------------"
# DiPLS modeling
DiPLS_model = DiPLS(s, p=l)
params = DiPLS_model.train(xk_mode1_train_scale, yk_mode1_train_scale)
pred_y3 = DiPLS_model.predict(params, xk_mode3_test_scale)

W_DiPLS = params['W']
beta =params['beta']
print(beta)
RMSE3=np.sqrt(mean_squared_error(yk_mode3_test_scale[s:],pred_y3))  # RMSE
print('DiPLS_RMSE:',RMSE3)

R3=r2_score(yk_mode3_test_scale[s:],pred_y3)
print('DiPLS_R2:',R3)


"--------------------------di-pls model--------------------------------------------------------"
#di-pls
model1 = ml1.model(xk_mode1_train_scale, yk_mode1_train_scale, xk_mode1_train_scale, xk_mode3_test, l)
l1=10
W_dipls = model1.train(l1)
pred_y4, _ = model1.predict(xk_mode3_test_scale, [])

RMSE4=np.sqrt(mean_squared_error(yk_mode3_test_scale[s:],pred_y4[s:]))  # RMSE
print('di-pls_RMSE:',RMSE4)

R4=r2_score(yk_mode3_test_scale[s:],pred_y4[s:])
print('di-pls_R2:',R4)

"------------------------Dynamic-di-pls model--------------------------------------------------"
#构建延迟矩阵
xk_mode1_train_scale_aug = augmented_matrix(xk_mode1_train_scale,s+1,xk_mode1_train_scale.shape[0]-s-1)
yk_mode1_train_scale_aug = yk_mode1_train_scale[s:yk_mode1_train_scale.shape[0]]

xk_mode3_test_scale_aug = augmented_matrix(xk_mode3_test_scale,s+1,xk_mode3_test_scale.shape[0]-s-1)
yk_mode3_test_scale_aug = yk_mode3_test_scale[s:yk_mode3_test_scale.shape[0]]

model11 = ml1.model(xk_mode1_train_scale_aug, yk_mode1_train_scale_aug, xk_mode1_train_scale_aug, xk_mode3_test_scale_aug, l)
l2=10
W_Ddipls = model11.train(l2)
pred_y5, _ = model11.predict(xk_mode3_test_scale_aug, [])

RMSE5=np.sqrt(mean_squared_error(yk_mode3_test_scale[s:],pred_y5))  # RMSE
print('Dynamic-di-pls_RMSE:',RMSE5)

R5=r2_score(yk_mode3_test_scale[s:],pred_y5)
print('Dynamic-di-pls_R2:', R5)

"-------------------------DTLVR model----------------------------------------------------"
#DTLVM

model2 =ml.model(xk_mode1_train_scale, yk_mode1_train_scale, xk_mode1_train_scale, xk_mode3_train_scale, l, s+1)
l3 = 10
W_DTLVR, beta1, beta2= model2.train(l)
pred_y6, _ = model2.predict(xk_mode3_test_scale, [])
print(beta1)
print(beta2)
RMSE6=np.sqrt(mean_squared_error(yk_mode3_test_scale[s:],pred_y6))  # RMSE
print('DTLVM_RMSE:',RMSE6)

R6=r2_score(yk_mode3_test_scale[s:],pred_y6)
print('DTLVM_R2:', R6)


plt.figure(1, figsize=(7,2),dpi=300)
plt.grid()
plt.plot(yk_mode3_test_scale[s:],color='blue',linestyle='-', label='True Value')
# plt.plot(pred_y1[s:],color='green', linestyle='-', label='PLS')
# plt.plot(pred_y2,color='cyan', linestyle='-', label='DPLS')
plt.plot(pred_y3,color='orange',linestyle='-', label='DiPLS')
# plt.plot(pred_y4[s:],color='magenta', linestyle='-', label='di-PLS')
# plt.plot(pred_y5,color='yellow',linestyle='-', label='Dynamic-di-pls')
plt.plot(pred_y6,color='red',linestyle='-', label='DTLVR')
plt.xlabel('Number of Testing Samples')
plt.ylabel('Actual output')
plt.legend(loc='upper right')
plt.show()


plt.figure(2,figsize=(7,3),dpi=300)
# plt.plot(np.abs(pred_y1[s:]-yk_mode3_test_scale[s:]),'g-',label='PLS')
# plt.plot(np.abs(pred_y2-yk_mode3_test_scale[s:]),'c-',label='DPLS')
plt.plot(np.abs(pred_y3-yk_mode3_test_scale[s:]),color='m',linestyle='-',label='DiPLS')
# plt.plot(np.abs(pred_y4[s:]-yk_mode3_test_scale[s:]),'m-',label='di-PLS')
plt.plot(np.abs(pred_y5-yk_mode3_test_scale[s:]),color='blue',linestyle='-',label='Dynamic-di-PLS')
plt.plot(np.abs(pred_y6-yk_mode3_test_scale[s:]),'r-',label='DTLVR')
plt.title('Prediction error comparison')
plt.legend(loc='upper right')
plt.show()


# plt.figure(3, figsize=(8,3),dpi=300)
# sns.kdeplot(xk_mode1_test[:,0],label='Source domain', shade=True)
# sns.kdeplot(xk_mode3_test[:,0],label='Target domain',shade=True)
# plt.legend()
# plt.show()

# print(W_dipls.shape)
from mpl_toolkits.mplot3d import Axes3D
fig4 = plt.figure(4, figsize=(7,5),dpi=300)
# P1 = np.array([[0.5586,0.2042,0.6370],[0.2007,0.0492,0.4429],[0.0874,0.6062,0.0664],[0.9332,0.5463,0.3743],[0.2594,0.0958,0.2491]])
tk_mode1_test = xk_mode1_test_scale@W_DTLVR
tk_mode3_test = xk_mode3_test_scale@W_DTLVR
ax = fig4.gca(projection='3d')
ax.scatter(tk_mode1_test[:,0],tk_mode1_test[:,1],tk_mode1_test[:,2])
ax.scatter(tk_mode3_test[:,0],tk_mode3_test[:,1],tk_mode3_test[:,2])
plt.show()


plt.figure(5, figsize=(8,3),dpi=300)
# sns.kdeplot(xk_mode1_test_scale[:,0],label='Source domain1', shade=True)
# sns.kdeplot(xk_mode3_test_scale[:,0],label='Target domain1',shade=True)
sns.kdeplot(tk_mode1_test[:,0],label='Source domain', shade=True)
sns.kdeplot(tk_mode3_test[:,0],label='Target domain',shade=True)
plt.legend()
plt.show()

plt.figure(6, figsize=(8,3),dpi=300)
tk_mode1_test = xk_mode1_train_scale@W_DTLVR
tk_mode3_test = xk_mode3_test_scale@W_DTLVR
plt.scatter(tk_mode1_test[:,0],tk_mode1_test[:,1])
plt.scatter(tk_mode3_test[:,0],tk_mode3_test[:,1])
plt.show()

beta1=[[0.92501805,0.0557145,-0.37581578],[0.80649749,0.00164455,-0.59123523],[0.8064527,-0.02717176,-0.59067398]]
