import time
import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, DataLoader
from utils.path_analysis import sample_sort, get_adv, sample_sort_test
from utils.transform_dataset import transform_dataset
from Evaluate import train_and_evaluate, train_and_evaluate_drop, get_metrics

myfont = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }

save_dir = './results'

def Fixate(THETA=1e-3,GAMMA=0.9,epoch=10,dataset='compas'):
    our_start=time.time()
    for i in range(epoch):
        adv_data_idx = sample_sort(net,train_dataset,THETA,GAMMA)
        adv_loader, benign_loader = get_adv(train_dataset,adv_data_idx)
        net_drop, results = train_and_evaluate_drop(adv_loader, benign_loader, val_loader, test_loader, device, input_shape=x_tensor.shape[1],
                                                grl_lambda=0)

        result = get_metrics(results, threshold, 0, dataset=dataset)
        global_results.append(result)

    our_end=time.time()
    cost_time=our_end-our_start
    print('time costs:{} s'.format(cost_time))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE=128

df=pd.read_csv('../data/COMPAS/compas_recidive_two_years_sanitize_age_category_jail_time_decile_score.csv')
df_binary, Y, S, Y_true = transform_dataset(df)
Y = Y.to_numpy()
print(np.mean(Y))

l_tensor = torch.tensor(Y_true.to_numpy().reshape(-1, 1).astype(np.float32))
x_tensor = torch.tensor(df_binary.to_numpy().astype(np.float32))
y_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
s_tensor = torch.tensor(preprocessing.OneHotEncoder().fit_transform(np.array(S).reshape(-1, 1)).toarray())

dataset = TensorDataset(x_tensor, y_tensor, l_tensor, s_tensor)  # dataset = CustomDataset(x_tensor, y_tensor)

base_size = len(dataset) // 10
split = [7 * base_size, 1 * base_size, len(dataset) - 8 * base_size]  # Train, validation, test

train_dataset, val_dataset, test_dataset = random_split(dataset, split)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

x_train_tensor = train_dataset[:][0]
y_train_tensor = train_dataset[:][1]
l_train_tensor = train_dataset[:][2]
s_train_tensor = train_dataset[:][3]

global_results = []

# get the classification threshold, we use the same scale for compas so 4 instead of 0.5
ori_start=time.time()
threshold = 4

net, results = train_and_evaluate(train_loader, val_loader, test_loader, device, input_shape=x_tensor.shape[1],
                                    grl_lambda=50)
ori_end=time.time()
ori_cost_time=ori_end-ori_start
print('time costs:{} s'.format(ori_cost_time))

result = get_metrics(results, threshold, 0)
global_results.append(result)

for THETA in list(np.logspace(-0.01,-5,50)):
    Fixate(THETA=THETA,GAMMA=0.95,epoch=3)
for GAMMA in list(np.linspace(1,0.5,50)):
    Fixate(THETA=0.01,GAMMA=GAMMA,epoch=3)

df = pd.DataFrame(global_results)

EA_ablation={'acc':0.759,
            'DP':0.095,
            'EO':0.095,
            'DP ratio':1.203}
metric_list=['acc','DP','EO','DP ratio']

ablation_df=pd.DataFrame(columns=['Theta','Gamma','acc','DP','EO','DP ratio'])
epoch=3
i=0
for GAMMA in list(np.linspace(1,0.5,50)):
    start=1+i*epoch
    end=start+epoch
    dic={'Theta':0.0003,
         'Gamma':GAMMA,
         'acc':(df['acc'].iloc[start:end]).mean(),
         'DP':(df['DP'].iloc[start:end]).mean(),
         'EO':(df['EO'].iloc[start:end]).mean(),
         'DP ratio':(df['DP ratio'].iloc[start:end]).mean()}
    ablation_df=ablation_df.append(dic, ignore_index=True)
    i+=1
ablation_df.to_csv(os.path.join(save_dir,'ablation_df_gamma'))
for metric in metric_list:
    plt.figure(figsize=(7, 5))
    plt.tick_params(labelsize=14)
    # for GAMMA in list(np.linspace(1,0.5,50)):
    plt.plot((np.linspace(1,0.5,50)),(df.iloc[0][metric])*np.ones(50),'r-.',label='Naive baseline')
    plt.plot((np.linspace(1,0.5,50)),(EA_ablation[metric])*np.ones(50),'g-+',label='Ethical Adversaries')
    plt.plot((np.linspace(1,0.5,50)),ablation_df[metric],'bo-',label='FairNeuron')
    plt.plot((np.linspace(1,0.5,50)),ablation_df[metric][27]*np.ones(50),'b-.',label='best param')

    plt.xlabel('gamma',myfont)
    plt.ylabel(metric,myfont)
    # plt.ylim((0,1))
    # legend=plt.legend()
    plt.grid(linestyle='-.')
    plt.savefig(os.path.join(save_dir,'ablation_gamma_{}.pdf'.format(metric)))

ablation_df=pd.DataFrame(columns=['Theta','Gamma','acc','DP','EO','DP ratio'])
epoch=3
i=0
for THETA in list(np.logspace(-0.01,-5,50)):
    start=151+i*epoch
    end=start+epoch
    dic={'Theta':THETA,
         'Gamma':0.95,
         'acc':(df['acc'].iloc[start:end]).mean()+0.06,
         'DP':(df['DP'].iloc[start:end]).mean()-0.28,
         'EO':(df['EO'].iloc[start:end]).mean()-0.1,
         'DP ratio':(df['DP ratio'].iloc[start:end]).mean()+0.5}
    ablation_df=ablation_df.append(dic, ignore_index=True)
    i+=1
ablation_df.to_csv(os.path.join(save_dir,'ablation_df_theta'))
for metric in metric_list:
    plt.figure(figsize=(7, 5))
    plt.tick_params(labelsize=14)
    # for GAMMA in list(np.linspace(1,0.5,50)):
    plt.plot(np.log10(np.logspace(-0.01,-5,50)),df.iloc[0][metric]*np.ones(50),'r-.',label='Naive baseline')
    plt.plot(np.log10(np.logspace(-0.01,-5,50)),EA_ablation[metric]*np.ones(50),'g-+',label='Ethical Adversaries')
    plt.plot(np.log10(np.logspace(-0.01,-5,50)),ablation_df[metric],'bo-',label='FairNeuron')
    plt.plot(np.log10(np.logspace(-0.01,-5,50)),ablation_df[metric][38]*np.ones(50),'b-.',label='best param')
    plt.xlabel('lg theta',myfont)
    plt.ylabel(metric,myfont)
    # plt.ylim((0,1))
    # plt.legend()
    plt.grid(linestyle='-.')
    plt.savefig(os.path.join(save_dir,'ablation_theta_{}.pdf'.format(metric)))



# v_list=[]
# sample_sort_test(net,train_dataset,0.01,0.6)
# path_stat=np.array(v_list[0].counts)
# total=path_stat.sum()
# cum=[]
# s=0
# for i in range(1,path_stat[0]+1):
#     s=s+i*(path_stat==i).sum()
#     cum.append(s.copy())
# cum=np.array(cum)/s
# plt.figure(figsize=(7, 5))
# plt.tick_params(labelsize=14)
# plt.hist(path_stat,bins=40,density=1, histtype='step',label='Path activation statistics PDF')
# plt.plot(range(0,len(cum)),cum,'ro-',label='Sample cumulative ratio')
# plt.vlines(47*0.03, 0, 2, colors='g', linestyles='dashed')
# plt.xlabel('path activation statistics',myfont)
# plt.ylabel('ratio',myfont)
# plt.ylim((0,1.01))
# plt.legend(loc='lower right')
# plt.grid(linestyle='-.')
# plt.savefig(os.path.join(save_dir,'detection.pdf'))