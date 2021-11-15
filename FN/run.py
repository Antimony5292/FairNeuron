import argparse
import torch
import time
import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch
from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import dataset
from torch.utils.data.dataset import random_split

from utils.transform_dataset import transform_dataset,transform_dataset_credit,transform_dataset_census
from Evaluate import get_metrics, train_and_evaluate
from FairNeuron import Fixate_with_val


class DataClass():
    def __init__(self,df,dataset) -> None:

        if dataset=='compas':
            df_binary, Y, S, Y_true = transform_dataset(df)
            Y = Y.to_numpy()    
            self.l_tensor = torch.tensor(Y_true.to_numpy().reshape(-1, 1).astype(np.float32))
            self.threshold=4
        elif dataset=='credit':
            df_binary, Y, S, Y_true = transform_dataset_credit(df)
            self.l_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
            self.threshold=0.5
        else:
            df_binary, Y, S, Y_true = transform_dataset_census(df)
            self.l_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
            self.threshold=0.5
        self.x_tensor = torch.tensor(df_binary.to_numpy().astype(np.float32))
        self.y_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
        self.s_tensor = torch.tensor(preprocessing.OneHotEncoder().fit_transform(np.array(S).reshape(-1, 1)).toarray())
        self.dataset = TensorDataset(self.x_tensor, self.y_tensor, self.l_tensor, self.s_tensor)
        base_size = len(self.dataset) // 10
        split = [7 * base_size, 1 * base_size, len(self.dataset) - 8 * base_size]  # Train, validation, test

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, split)
        self.x_train_tensor = self.train_dataset[:][0]
        self.y_train_tensor = self.train_dataset[:][1]
        self.l_train_tensor = self.train_dataset[:][2]
        self.s_train_tensor = self.train_dataset[:][3]
        self.global_results=[]
        


def run(dataset,inputpath,outputpath,epoch,BATCH_SIZE):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE=128
    file_name='{}_epoch{}_{}'.format(dataset,epoch,int(time.time()))
    print(os.path.join(outputpath,file_name))
    if dataset=='credit':
        df=pd.read_csv(inputpath,sep=' ')
    else:
        df=pd.read_csv(inputpath)
    data_class = DataClass(df,dataset)


    # dataset = TensorDataset(x_tensor, y_tensor, l_tensor, s_tensor)  # dataset = CustomDataset(x_tensor, y_tensor)

    # base_size = len(dataset) // 10
    # split = [7 * base_size, 1 * base_size, len(dataset) - 8 * base_size]  # Train, validation, test

    # train_dataset, val_dataset, test_dataset = random_split(dataset, split)

    train_loader = DataLoader(dataset=data_class.train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(dataset=data_class.val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset=data_class.test_dataset, batch_size=BATCH_SIZE)

    # x_train_tensor = train_dataset[:][0]
    # y_train_tensor = train_dataset[:][1]
    # l_train_tensor = train_dataset[:][2]
    # s_train_tensor = train_dataset[:][3]

    # global_results = []

    # get the classification threshold, we use the same scale for compas so 4 instead of 0.5
    ori_start=time.time()
    threshold = 4

    net, results = train_and_evaluate(train_loader, val_loader, test_loader, device, input_shape=data_class.x_tensor.shape[1],
                                        grl_lambda=0)


    ori_end=time.time()
    ori_cost_time=ori_end-ori_start
    print('time costs:{} s'.format(ori_cost_time))

    result = get_metrics(results, threshold, 0)
    data_class.global_results.append(result)
    net_nodrop, results = train_and_evaluate(train_loader, val_loader, test_loader, device, input_shape=data_class.x_tensor.shape[1],
                                        grl_lambda=0,dataset='compas_nodrop')
    result = get_metrics(results, threshold, 0)
    data_class.global_results.append(result)

    # EA
    # EA(net,attack_size=10, iter_num=50)

    Fixate_with_val(net,data_class,epoch=epoch,BATCH_SIZE=BATCH_SIZE)

    res = pd.DataFrame(data_class.global_results)
    res.to_csv(os.path.join(outputpath,file_name))

if __name__ == '__main__':
    # torch.cuda.set_device(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices={'compas','census','credit'},default='compas')
    parser.add_argument('--epoch',default=10)
    parser.add_argument('--batch-size',default=128,dest='batchsize')
    parser.add_argument('--input-path',default='../data/COMPAS/compas_recidive_two_years_sanitize_age_category_jail_time_decile_score.csv',dest='inputpath')
    parser.add_argument('--save-dir',default='./results',dest='outputpath')
    parser.add_argument('--rand',action='store_true')
    args=parser.parse_args()

    run(dataset=args.dataset,inputpath=args.inputpath,outputpath=args.outputpath,epoch=args.epoch,BATCH_SIZE=args.batchsize)
