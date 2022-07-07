import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pycm import ConfusionMatrix
from sklearn import preprocessing
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, DataLoader
from ray import tune
from FN.utils.transform_dataset import transform_dataset,transform_dataset_census,transform_dataset_credit
from FN.Evaluate import bm,train_and_evaluate

def get_metrics_ROC(results, threshold, margin, fraction=0,dataset='compas',privileged=1,unprivileged=0):
    "Create the metrics from an output df."

    # Calculate biases after training
    dem_parity = abs(
        bm(results).P(pred=lambda x: x > threshold-margin).given(race=privileged)
        - bm(results).P(pred=lambda x: x > threshold+margin).given(
            race=unprivileged))

    eq_op = abs(
        bm(results).P(pred=lambda x: x > threshold-margin).given(race=privileged, compas=True)
        - bm(results).P(pred=lambda x: x > threshold+margin).given(race=unprivileged, compas=True))

    dem_parity_ratio = abs(
        bm(results).P(pred=lambda x: x > threshold-margin).given(race=privileged)
        / bm(results).P(pred=lambda x: x > threshold+margin).given(
            race=unprivileged))
    correct = ((results['true']) == (results['pred'] > threshold)).sum()-((results['pred'] < threshold+margin) & (results['pred'] > threshold-margin) & (results['race'] == 1) &
                                                     (results['true'] == False)).sum()-((results['pred'] < threshold+margin) & (results['pred'] > threshold-margin) & (results['race'] == 0) & (results['true'] == True)).sum()
    roc_acc = correct / len(results)
#     roc_acc = bm(results).P(pred=lambda x: x > threshold+margin).given(true>threshold)
    cm = ConfusionMatrix(actual_vector=(results['true'] == True).values,
                         predict_vector=(results['pred'] > threshold).values)
    if dataset=='compas':
        cm_high_risk = ConfusionMatrix(actual_vector=(results['compas'] > 8).values,
                             predict_vector=(results['pred'] > 8).values)
    
        result = {"DP": dem_parity,
                  "EO": eq_op,
                  "DP ratio": dem_parity_ratio,
                  "roc_acc":roc_acc,
                  "acc": cm.Overall_ACC,
                  "acc_ci_min": cm.CI95[0],
                  "acc_ci_max": cm.CI95[1],
                  "f1": cm.F1_Macro,
                  "acc_high_risk": cm_high_risk.Overall_ACC,
                  "acc_ci_min_high_risk": cm_high_risk.CI95[0],
                  "acc_ci_max_high_risk": cm_high_risk.CI95[1],
                  "f1_high_risk": cm_high_risk.F1_Macro,
                  "adversarial_fraction": fraction
                  }
    else:
        result = {"DP": dem_parity,
                  "EO": eq_op,
                  "DP ratio": dem_parity_ratio,
                  "roc_acc":roc_acc,
                  "acc": cm.Overall_ACC,
                  "acc_ci_min": cm.CI95[0],
                  "acc_ci_max": cm.CI95[1],
                  "f1": cm.F1_Macro,
                  "adversarial_fraction": fraction
                  }

    return result

def training_function_ROC(config):
    
    margin = config['margin']
    train_loader_s=config['train']
    val_loader_s=config['val']
    test_loader_s=config['test']
    x_train_tensor_s=config['x_tensor']


    net_drop, results = train_and_evaluate(train_loader_s, val_loader_s, test_loader_s, device, input_shape=x_train_tensor_s.shape[1],
                                            grl_lambda=50,dataset=config['dataset'])
    result = get_metrics_ROC(results, threshold, margin,dataset=config['dataset'])
    print(result)
    complex_score = result['DP']+result['EO']+(1-result['DP ratio'])-0.01*result['acc']
    tune.report(mean_loss=complex_score)


def ROC_evaluate(dataset='compas'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE=128    
    if dataset=='compas':
        df=pd.read_csv('data/COMPAS/compas_recidive_two_years_sanitize_age_category_jail_time_decile_score.csv')
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

        base_size = len(train_dataset) // 10
        split = [8 * base_size, len(train_dataset) - 8 * base_size]  # Train, validation, test
        train_dataset_s, val_dataset_s = random_split(train_dataset, split)
        test_dataset_s = val_dataset
        #     print(train_dataset_s)

        #     train_loader_s = DataLoader(dataset=train_dataset_s, batch_size=BATCH_SIZE, shuffle=True)
        train_loader_s = DataLoader(dataset=train_dataset_s, batch_size=BATCH_SIZE)
        val_loader_s = DataLoader(dataset=val_dataset_s, batch_size=BATCH_SIZE)
        test_loader_s = DataLoader(dataset=test_dataset_s, batch_size=BATCH_SIZE)

        x_train_tensor_s = val_dataset[:][0]


        analysis = tune.run(
            training_function_ROC,
            config={
                'margin': tune.grid_search(list(np.linspace(0,0.25,26))),
                'dataset':'compas',
                'train':train_loader_s,
                'val':val_loader_s,
                'test':test_loader_s,
                'x_tensor':x_train_tensor_s
            },
            resources_per_trial={
                "cpu": 16,
                "gpu": 2,
            }
        )
        best_config=analysis.get_best_config(metric="mean_loss", mode="min")
        print("Best config: ",best_config)
        margin = best_config['margin']

        net, results = train_and_evaluate(train_loader, val_loader, test_loader, device, input_shape=x_tensor.shape[1],
                                            grl_lambda=50)
        result = get_metrics_ROC(results, threshold, margin,dataset='compas')
    elif dataset=='census':
        df=pd.read_csv('data/Census/adult')
        df_binary, Y, S, Y_true = transform_dataset_census(df)
        print(np.mean(Y))

        l_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
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
        threshold = 0.5

        base_size = len(train_dataset) // 10
        split = [8 * base_size, len(train_dataset) - 8 * base_size]  # Train, validation, test
        train_dataset_s, val_dataset_s = random_split(train_dataset, split)
        test_dataset_s = val_dataset
        #     print(train_dataset_s)

        #     train_loader_s = DataLoader(dataset=train_dataset_s, batch_size=BATCH_SIZE, shuffle=True)
        train_loader_s = DataLoader(dataset=train_dataset_s, batch_size=BATCH_SIZE)
        val_loader_s = DataLoader(dataset=val_dataset_s, batch_size=BATCH_SIZE)
        test_loader_s = DataLoader(dataset=test_dataset_s, batch_size=BATCH_SIZE)

        x_train_tensor_s = val_dataset[:][0]


        analysis = tune.run(
            training_function_ROC,
            config={
                'margin': tune.grid_search(list(np.linspace(0,0.25,26))),
                'dataset':'census',
                'train':train_loader_s,
                'val':val_loader_s,
                'test':test_loader_s,
                'x_tensor':x_train_tensor_s
            },
            resources_per_trial={
                "cpu": 16,
                "gpu": 2,
            }
        )
        best_config=analysis.get_best_config(metric="mean_loss", mode="min")
        print("Best config: ",best_config)
        margin = best_config['margin']

        net, results = train_and_evaluate(train_loader, val_loader, test_loader, device, input_shape=x_tensor.shape[1],
                                            grl_lambda=50,dataset='census')



        result = get_metrics_ROC(results, threshold, margin,dataset='census')
    elif dataset=='credit':
        df=pd.read_csv('data/Credit/german_credit',sep=' ')
        df_binary, Y, S, Y_true = transform_dataset_credit(df)
        print(np.mean(Y))

        l_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
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
        threshold = 0.5

        base_size = len(train_dataset) // 10
        split = [8 * base_size, len(train_dataset) - 8 * base_size]  # Train, validation, test
        train_dataset_s, val_dataset_s = random_split(train_dataset, split)
        test_dataset_s = val_dataset
        #     print(train_dataset_s)

        #     train_loader_s = DataLoader(dataset=train_dataset_s, batch_size=BATCH_SIZE, shuffle=True)
        train_loader_s = DataLoader(dataset=train_dataset_s, batch_size=BATCH_SIZE)
        val_loader_s = DataLoader(dataset=val_dataset_s, batch_size=BATCH_SIZE)
        test_loader_s = DataLoader(dataset=test_dataset_s, batch_size=BATCH_SIZE)

        x_train_tensor_s = val_dataset[:][0]


        analysis = tune.run(
            training_function_ROC,
            config={
                'margin': tune.grid_search(list(np.linspace(0,0.25,26))),
                'dataset':'census',
                'train':train_loader_s,
                'val':val_loader_s,
                'test':test_loader_s,
                'x_tensor':x_train_tensor_s
            },
            resources_per_trial={
                "cpu": 16,
                "gpu": 2,
            }
        )
        best_config=analysis.get_best_config(metric="mean_loss", mode="min")
        print("Best config: ",best_config)
        margin = best_config['margin']

        net, results = train_and_evaluate(train_loader, val_loader, test_loader, device, input_shape=x_tensor.shape[1],
                                            grl_lambda=50,dataset='credit')



        result = get_metrics_ROC(results, threshold, margin,dataset='credit')
    return result