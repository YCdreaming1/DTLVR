import numpy as np
import matplotlib.pyplot as plt
import functions as fct
import math
import scipy.io as sio
import seaborn as sns
import scipy
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
np.random.seed(5)
Num_data = 631
import random


def gengaus(length, mu,sigma, mag,noise=0):
    '''
        Generate a spectrum-like Gaussian signal with random noise

        Params
        ------

        length: int
            Length of the signal (i.e. number of variables)

        mu: float
            Mean of the Gaussian signal

        sigma: float
            Standard deviation of the Gaussian

        mag: float
            Magnitude of the signal

        noise: float
            Amount of i.i.d noise


        Returns
        -------

        signal: numpy array (length x 1)
            Generated Gaussian signal
        '''

    s = mag * scipy.stats.norm.pdf(np.arange(length), mu, sigma)
    n = noise * np.random.rand(length)
    signal = s + n

    return signal


"系统变换矩阵"
P1 = np.array([[0.5586,0.2042,0.6370],[0.2007,0.0492,0.4429],[0.0874,0.6062,0.0664],[0.9332,0.5463,0.3743],[0.2594,0.0958,0.2491]])

"输入白噪声"
ek = np.random.normal(0,0.01, [Num_data,5])

"输出白噪声"
vk = np.random.normal(0,0.01, [Num_data-1,1])

"回归系数矩阵"
C1 = np.array([0.7451, 0.4928, 0.7320, 0.4738, 0.5652]).reshape((5,1))
C2 = np.array([1.9939, 0.7728, 1.0146, 1.1563, 1.2307]).reshape((5,1))

"系统延迟矩阵"
A1 = np.array([[0.4389,0.1210,-0.0862],[-0.2966, -0.1550,0.2274],[0.3538,-0.6573,0.4239]])
A2 = np.array([[-0.2998,-0.1905,-0.2669],[-0.0204,-0.1585,-0.2950],[0.1461,-0.0755,0.3749]])
fk = np.random.normal(0,0.01,[Num_data+2, 3])

"生成多模态数据集 Multi-grade Simulation Data"
"每个模态分别生成300个数据"

"模态1"
tks_mode1 = np.zeros((Num_data+2,3))
tk_mode1 = np.random.multivariate_normal([-1.3,-1.7,-1.1],[[3.5,1.2,1.89],[1.2,1.26,0.46],[1.89,0.46,0.64]], Num_data+2)
tks_mode1[0,:]=tk_mode1[0,:]
tks_mode1[1,:]=tk_mode1[1,:]

for i in range(Num_data+2):
    if i >1:
        tks_mode1[i, :] = (A1@tks_mode1[i-1, :].T-A2@tks_mode1[i-2,:].T+tk_mode1[i,:].T).T
tk_mode1= tks_mode1[2:,:]+fk[2:,:]

# tk_mode1 = tk_mode1.T

xk_mode1 = (P1@tk_mode1.T+ek.T).T
yk_mode1 = (C1.T@xk_mode1[1:Num_data,:].T+C2.T@xk_mode1[0:Num_data-1,:].T).T+vk

# 保存数据
'python格式'
np.save(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\xk_mode1.npy',xk_mode1[31:])
np.save(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\tk_mode1.npy',tk_mode1[31:])
np.save(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\yk_mode1.npy',yk_mode1[30:])

'Matlab格式'
sio.savemat(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\xk_mode1.mat',{'xk_mode1':xk_mode1[31:]})
sio.savemat(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\tk_mode1.mat',{'tk_mode1':tk_mode1[31:]})
sio.savemat(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\yk_mode1.mat',{'yk_mode1':yk_mode1[30:]})



"模态2"
tks_mode2 = np.zeros((Num_data+2,3))
tk_mode2 = np.random.multivariate_normal([1,-2,1.2],[[1.49,2.3,-0.7],[2.3,1.25,-0.96],[-0.7,-0.96,0.66]],Num_data+2)
tks_mode2[0,:]=tk_mode2[0,:]
tks_mode2[1,:]=tk_mode2[1,:]

A1 = np.array([[0.2334,-0.0836,0.2573],[0.1783, -0.1590,0.3416],[0.4345,-0.4573,0.3239]])
A2 = np.array([[-0.1998,-0.0905,-0.2669],[-0.2141,-0.1585,-0.3950],[-0.1461,0.6755,-0.3749]])


for i in range(Num_data+2):
    if i >1:
        tks_mode2[i, :] = (A1@tks_mode2[i-1, :].T-A2@tks_mode2[i-2,:].T+tk_mode2[i,:].T).T
tk_mode2 = tks_mode2[2:,:]+fk[2:,:]
# tk_mode2 = tk_mode2.T

# C1 = np.array([0.2334, 0.5668, -1.4485, 1.4738, 1.5652]).reshape((5,1))
# C2 = np.array([0.9939, 0.3842, -0.0146, 1.1563, 1.5845]).reshape((5,1))

xk_mode2 = (P1@tk_mode2.T+ek.T).T
yk_mode2 = (C1.T@xk_mode2[1:Num_data,:].T+C2.T@xk_mode2[0:Num_data-1,:].T).T+vk

# 保存数据
'python格式'
np.save(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\xk_mode2.npy',xk_mode2[31:])
np.save(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\tk_mode2.npy',tk_mode2[31:])
np.save(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\yk_mode2.npy',yk_mode2[30:])

'Matlab格式'
sio.savemat(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\xk_mode2.mat',{'xk_mode2':xk_mode2[31:]})
sio.savemat(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\tk_mode2.mat',{'tk_mode2':tk_mode2[31:]})
sio.savemat(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\yk_mode2.mat',{'yk_mode2':yk_mode2[30:]})



"模态3"
# tk_1=np.array([1.2163, 1.17, 0.8356])
# tk_2=np.array([1.0191, 0.1009,0.7900])
tks_mode3 = np.zeros((Num_data+2,3))
tk_mode3 = np.random.multivariate_normal([1,0,3],[[3.4,1.3,0.66],[1.3,1.2,-0.05],[0.66,-0.05,0.78]], Num_data+2)
tks_mode3[0,:]=tk_mode3[0,:]
tks_mode3[1,:]=tk_mode3[1,:]
# plt.plot(tk_mode3)
# plt.show()



A1 = np.array([[0.2389,0.1210,-0.0862],[-0.2966, -0.0550,0.2274],[0.0154,-0.6573,0.0423]])
A2 = np.array([[-0.2998,-0.1905,-0.2669],[-0.0204,-0.1585,-0.2950],[0.1461,-0.0755,0.3749]])



for i in range(Num_data+2):
    if i >1:
        tks_mode3[i, :] = (A1@tks_mode3[i-1, :].T-A2@tks_mode3[i-2,:].T+tk_mode3[i,:].T).T
tk_mode3 = tks_mode3[2:,:]+fk[2:,:]
# tk_mode3 = tk_mode3.T
xk_mode3 = (P1@tk_mode3.T+ek.T).T
yk_mode3 = (C1.T@xk_mode3[1:Num_data,:].T+C2.T@xk_mode3[0:Num_data-1,:].T).T+vk

# 保存数据
'python格式'
np.save(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\xk_mode3.npy',xk_mode3[31:])
np.save(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\tk_mode3.npy',tk_mode3[31:])
np.save(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\yk_mode3.npy',yk_mode3[30:])

'Matlab格式'
sio.savemat(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\xk_mode3.mat',{'xk_mode3':xk_mode3[31:]})
sio.savemat(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\tk_mode3.mat',{'tk_mode3':tk_mode3[31:]})
sio.savemat(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\yk_mode3.mat',{'yk_mode3':yk_mode3[30:]})


from mpl_toolkits.mplot3d import Axes3D
plt.figure(1, figsize=(8,2),dpi=300)
plt.plot(np.vstack([yk_mode1[30:],yk_mode2[30:],yk_mode3[30:]]))
# plt.plot(yk_mode2[30:],'b-',label='Mode3')
# plt.legend(loc='upper right')
# print(yk_mode3[30:].shape)
plot_acf(yk_mode2[30:,0]).show()
fig = plt.figure(figsize=(7,5),dpi=300)
ax = fig.gca(projection='3d')
ax.scatter(tk_mode1[:,0],tk_mode1[:,1],tk_mode1[:,2],label='mode1')
ax.scatter(tk_mode3[:,0],tk_mode3[:,1],tk_mode3[:,2],label='mode2')
ax.scatter(tk_mode2[:,0],tk_mode2[:,1],tk_mode2[:,2],label='mode3')

sio.savemat(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\tk_mode1.mat',{'tk_mode1':tk_mode1})
sio.savemat(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\tk_mode2.mat',{'tk_mode2':tk_mode2})
sio.savemat(r'C:\Users\youngc\PycharmProjects\IndustrialModeling\Multimode_Numerical example\tk_mode3.mat',{'tk_mode3':tk_mode3})
plt.xlabel('The first LV')
plt.ylabel('The second LV')
ax.set_zlabel('The third LV')
plt.legend(loc='best')
plt.show()



tka = np.random.normal([4.1,2.7,6.1], [2,2,2], [Num_data, 3])
tkb = np.random.normal([-2,-4,-1], [1,2.3,0.8], [Num_data, 3])
tkc=tka+tkb

plt.figure(2, figsize=(8,2), dpi=300)
sns.kdeplot(tka[:,0], shade=True)
sns.kdeplot(tkb[:,0], shade=True)
plt.show()


plt.figure(3, figsize=(8,2), dpi=300)
plt.plot(tk_mode2)
plt.title('Mode3: three input LVs')
plt.show()








