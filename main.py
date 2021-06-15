from sklearn.cluster import KMeans
from munkres import Munkres,print_matrix     
import glob
from sklearn.metrics.cluster import normalized_mutual_info_score
import pandas as pd
import os.path as osp
import os
import numpy as np

root_path = ##your path
dataset = sorted(os.listdir(root_path))

x_1 = pd.read_csv(osp.join(root_path,dataset[0]),header=None).to_numpy()
print(x_1)
y_1 = pd.read_csv(osp.join(root_path,dataset[1]),header=None).to_numpy()
x_1.shape

Final_result['GLIOMA'][20][0]

from sklearn.metrics.cluster import normalized_mutual_info_score
np_ = len(np.unique(y_1))
# new_x = x_1[:,[6525,  584, 6129,  518, 1605, 6061, 4250, 5885, 1625,  483,  536,431,  716, 4896, 6161, 3397, 1993,  726, 4510,  589]]
idx_1 = pd.DataFrame(KMeans(n_clusters=np_).fit_predict(new_x))
idx_ = idx_1[0].to_numpy()+1

y_1 = pd.read_csv(osp.join(root_path,dataset[1]))
Y_ = y_1['1'].to_numpy()
Y_
print(ClusterAccMea(Y_,idx_))
normalized_mutual_info_score(Y_,idx_)

np.arange(9)*2

## RNE experiment

import pandas as pd
import os.path as osp
import os
import numpy as np
from sklearn.cluster import KMeans
from munkres import Munkres,print_matrix     
import glob
import sklearn
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import v_measure_score
import time

root_path = '/content/drive/MyDrive/colab_data/bigdata_course_data'
dataset = sorted(os.listdir(root_path))

#dataset_1 = ALLAML  dataset_2 = GLIOMA  dataset_3 = Isolet1
dataset_idx = np.arange(0,9)*2
H_idx_ = np.arange(1,11)*10

#RNE training
Final_result = {}
for idx, data in enumerate(dataset):
    if idx in dataset_idx:
        data_name = data.split('_')[0]
        dataset = pd.read_csv(osp.join(root_path,data),header=None)
        k = 5
        m_ = np.arange(20,101,10)
        result={}
        for i, m in enumerate(m_):
            H1, _ = RNE_obj(dataset,m,k)
            result[m] = H1
        Final_result[data_name] = result
print(Final_result)

#RNE evaluation
for data_idx in dataset_idx:
    data_name = dataset[data_idx].split('_')[0]
    dataset_x = pd.read_csv(osp.join(root_path,dataset[data_idx]),header=None)
    dataset_y = pd.read_csv(osp.join(root_path,dataset[data_idx+1]),header=None)
    X = dataset_x.to_numpy()
    X = sklearn.preprocessing.normalize(X)
    Y = dataset_y[0].to_numpy()
    np_ = len(np.unique(Y))
    acc_mean = np.zeros(9)
    acc_std = np.zeros(9)
    nmi_mean = np.zeros(9)
    nmi_std = np.zeros(9)
    time_ = np.zeros(9)
    print(f'start evaluate {data_name}')
    for j,H_idx in enumerate(H_idx_):
        temp_acc = np.zeros(30)
        temp_nmi = np.zeros(30)
        if j == 0:
            time_st = time.time()
            for i in range(30):
                idx = pd.DataFrame(KMeans(n_clusters=np_).fit_predict(X))
                idx_ = idx[0].to_numpy()
                temp_acc[i] = ClusterAccMea(Y,idx_)
                temp_nmi[i] = normalized_mutual_info_score(Y,idx_)
            time_fin = time.time()
            print(f'baseline acc mean = {np.mean(temp_acc)}')
            print(f'baseline acc std = {np.std(temp_acc)}')
            print(f'baseline nmi mean = {np.mean(temp_nmi)}')
            print(f'baseline nmi std = {np.std(temp_nmi)}')
            print(f'baseline evaluation time = {round((time_fin-time_st)/30,5)}')
        else:
            # result = LS_result[data_name][j-1]

            result = Final_result[data_name][H_idx] 
            if type(result) == list:
                I = np.asarray(result[0])-1
            else:
                I= result-1
            time_st = time.time()
            for i in range(30):
                idx = pd.DataFrame(KMeans(n_clusters=np_).fit_predict(X[:,I]))
                idx_ = idx[0].to_numpy()
                # y_permuted_predict = best_map(Y, idx_)
                # temp_acc[i] = accuracy_score(Y, y_permuted_predict)
                temp_acc[i] = ClusterAccMea(Y,idx_)
                # import pdb; pdb.set_trace()
                temp_nmi[i] = normalized_mutual_info_score(Y,idx_)
            time_fin = time.time()
            acc_mean[j-1] = np.mean(temp_acc)
            acc_std[j-1] = np.std(temp_acc)
            nmi_mean[j-1] = np.mean(temp_nmi)
            nmi_std[j-1] = np.std(temp_nmi)
            time_[j-1] = round((time_fin-time_st)/30,5)
            if j == 9:
                print(f'RNE acc mean = {np.max(acc_mean)}')
                max_idx_acc = np.argmax(acc_mean)
                print(f'RNE acc std = {acc_std[max_idx_acc]}')
                # print(acc_std)
                print(f'RNE nmi mean = {np.max(nmi_mean)}')
                max_idx_nmi = np.argmax(nmi_mean)
                print(f'RNE nmi std = {nmi_std[max_idx_nmi]}')
                print(f'RNE evaluation time = {time_[max_idx_acc]}')
                
                # print(nmi_std)
    print(f'finish evaluate {data_name}')
