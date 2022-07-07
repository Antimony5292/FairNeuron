import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import time
import numpy as np
# np.random.seed(0)
import pandas as pd
from sklearn import preprocessing
import torch
from aif360.datasets import GermanDataset,AdultDataset,CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from FN.Evaluate import train_and_evaluate,get_metrics

def transform_dataset_compas_reweight(df):
    """

    :param df:
    :return: Tuple of the transformed dataset and the labels Y and S
    """


    ##separated class from the rests of the features
    # remove unnecessary dimensions from Y -> only the decile_score remains
    Y = df['decile_score']
    del df['decile_score']
    Y_true = df['two_year_recid']
    del df['two_year_recid']

    S = df['race']
    #del df['race']
    #del df['is_recid']

    print(df.shape)

    # set sparse to False to return dense matrix after transformation and keep all dimensions homogeneous
    encod = preprocessing.OneHotEncoder(sparse=False)

    data_to_encode = df.to_numpy()
    feat_to_encode = data_to_encode[:, 0]
    # print(feat_to_encode)
    # transposition
    feat_to_encode = feat_to_encode.reshape(-1, 1)
    # print(feat_to_encode)
    encoded_feature = encod.fit_transform(feat_to_encode)

    df_encoded = pd.DataFrame(encoded_feature)

    feat_to_encode = data_to_encode[:, 1]
    feat_to_encode = feat_to_encode.reshape(-1, 1)
    encoded_feature = encod.fit_transform(feat_to_encode)


    df_encoded = pd.concat([df_encoded, pd.DataFrame(encoded_feature)], axis=1)

    feat_to_encode = data_to_encode[:, 2] == "Caucasian"
    feat_to_encode = feat_to_encode.reshape(-1, 1)
    encoded_feature = encod.fit_transform(feat_to_encode)

    df_encoded = pd.concat([df_encoded, pd.DataFrame(encoded_feature)], axis=1)

    # feature [2] [3] [4] [5] [6] [7] [8] has to be put between 0 and 1

    for i in range(3, 10):
        encoded_feature = data_to_encode[:, i]
        ma = np.amax(encoded_feature)
        mi = np.amin(encoded_feature)
        encoded_feature = (encoded_feature - mi) / (ma - mi)
        df_encoded = pd.concat([df_encoded, pd.DataFrame(encoded_feature)], axis=1)

    feat_to_encode = data_to_encode[:, 10]
    feat_to_encode = feat_to_encode.reshape(-1, 1)
    encoded_feature = encod.fit_transform(feat_to_encode)

    df_encoded = pd.concat([df_encoded, pd.DataFrame(encoded_feature)], axis=1)

    feat_to_encode = data_to_encode[:, 11]
    feat_to_encode = feat_to_encode.reshape(-1, 1)
    encoded_feature = encod.fit_transform(feat_to_encode)

    df_encoded = pd.concat([df_encoded, pd.DataFrame(encoded_feature)], axis=1)

    return df_encoded, Y, S, Y_true

def transform_dataset_census_reweight(df):
    """
    :param df: the dataset "census income" from a csv file with reduced features, heterogeneous types and missing values, no header
    :return: Tuple of the transformed dataset and the labels Y and S
    """
    df_replace = df.replace(to_replace="?",value=np.nan)
    df_replace.dropna(inplace=True, axis=0)

    label_encoder = preprocessing.LabelEncoder()
    oh_encoder = preprocessing.OneHotEncoder(sparse=False)

    df_label = df_replace.iloc[:,-1]

    ##Y_true is the vector containing labels, at this point, labels (initially strings) have been transformed into integer (0 and 1) -> -5000 is now '0' and 5000+ is now '+1'
    Y = label_encoder.fit_transform(df_label)
    #remove last column from df
    del df_replace[df_replace.columns[-1]]

    # Y_true is the true outcome, in this case we're not using a future predictor (vs. compas)
    Y_true=[]

    #S is the protected attribute
    # could also be feature 7 (sex) or feature 13 (citizenship)
    S=df_replace["sex"]
    del df_replace["sex"]

    #remove feature fnlwgt
#     del df_replace["fnlwgt"]

    #remove examples with missing values
              ## change 1 to 0 

    #     if df_replace.shape == df.shape:
    #         raise AssertionError("The removal of na values failed")

    print(df_replace.shape)

    #transform other features
    #feature age to normalize
    encoded_feature = df_replace.to_numpy()[:, 0]
    mi = np.amin(encoded_feature)
    ma = np.amax(encoded_feature)
    encoded_feature = (encoded_feature - mi) / (ma - mi)

    #df_binary_encoded is the data frame containing encoded features
    df_binary_encoded = pd.DataFrame(encoded_feature)
    print(df_binary_encoded.shape)


    encod_feature = df_replace.iloc[:,1]
    encoded_feature = pd.get_dummies(encod_feature)
    # df_binary_encoded = pd.concat([df_binary_encoded, pd.DataFrame(encoded_feature)], axis=1)

    #feature 1 to 7 (after removal) are categorical
    for i in range(1,8):
        encod_feature = df_replace.iloc[:,i]
    #     print(encod_feature.shape)
        encoded_feature = pd.get_dummies(encod_feature)
    #     print(encoded_feature)
    #     print(df_binary_encoded)
        df_binary_encoded = pd.concat([df_binary_encoded.reset_index(drop=True), pd.DataFrame(encoded_feature).reset_index(drop=True)], axis=1)
#         print(df_binary_encoded)
    #     print('')

    #feature 8 and 9 are numerical
    for i in range(8,10):
        encod_feature = df_replace.iloc[:,i]
        mi = np.amin(encod_feature)
        ma = np.amax(encod_feature)
        encoded_feature = (encod_feature - mi) / (ma - mi)
        df_binary_encoded = pd.concat([df_binary_encoded.reset_index(drop=True), pd.DataFrame(encoded_feature).reset_index(drop=True)], axis=1)
    #     print(df_binary_encoded.shape)
    #feature 10 and 11 are categorical
    for i in range(10,12):
        encod_feature = df_replace.iloc[:,i]
        encoded_feature = pd.get_dummies(encod_feature)
        df_binary_encoded = pd.concat([df_binary_encoded.reset_index(drop=True), pd.DataFrame(encoded_feature).reset_index(drop=True)], axis=1)
    #     print(df_binary_encoded.shape)

    return df_binary_encoded, Y, S, Y_true

def transform_dataset_credit_reweight(df):

    label_encoder = preprocessing.LabelEncoder()
    oh_encoder = preprocessing.OneHotEncoder(sparse=False)

    Y = np.array(df.iloc[:,-1] == 'Good Credit')

    del df[df.columns[-1]]

    # Y_true is the true outcome, in this case we're not using a future predictor (vs. compas)
    Y_true=[]

    #S is the protected attribute
    S=df.iloc[:,12] > 25
    #del df["Age"]

    #remove examples with missing values
    df_replace = df.replace(to_replace="?",value=np.nan)
    df_replace.dropna(inplace=True, axis=1)

    print(df_replace.shape)

    #transform other features
    #feature age to normalize
    encoded_feature = df_replace.to_numpy()[:, 1]
    mi = np.amin(encoded_feature)
    ma = np.amax(encoded_feature)
    encoded_feature = (encoded_feature - mi) / (ma - mi)

    #df_binary_encoded is the data frame containing encoded features
    df_binary_encoded = pd.DataFrame(encoded_feature)

    # categorical attributes
    for i in [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18,19]:
        encod_feature = df_replace.iloc[:,i]
        encoded_feature = pd.get_dummies(encod_feature)
        df_binary_encoded = pd.concat([df_binary_encoded, pd.DataFrame(encoded_feature)], axis=1)

    # Numerical attributes
    for i in [1, 7, 10, 15, 17]:
        encod_feature = df_replace.iloc[:,i]
        mi = np.amin(encod_feature)
        ma = np.amax(encod_feature)
        encoded_feature = (encod_feature - mi) / (ma - mi)
        df_binary_encoded = pd.concat([df_binary_encoded, pd.DataFrame(encoded_feature)], axis=1)

    print(S)

    return df_binary_encoded, Y, S, Y_true


def gen_preprocess_dataset(dataset='compas'):
    if dataset=='compas':
        dataset_orig = CompasDataset(
            features_to_drop=[],
            features_to_keep=['sex', 'age_cat', 'race',
                     'juv_fel_count', 'juv_misd_count', 'juv_other_count',
                     'priors_count', 'c_charge_degree',
                     'two_year_recid','decile_score','days_b_screening_arrest','c_jail_time (days)','date_dif_in_jail','is_recid']
        )

        dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]

        RW = Reweighing(unprivileged_groups=unprivileged_groups,
                        privileged_groups=privileged_groups)
        dataset_transf_train = RW.fit_transform(dataset_orig_train)
        df,_ = dataset_transf_train.convert_to_dataframe(de_dummy_code=True,set_category=False)
        test_df,_=dataset_orig_test.convert_to_dataframe(de_dummy_code=True,set_category=False)
        df.to_csv('./data/COMPAS/reweighting/reweighting_train',index=False)
        test_df.to_csv('./data/COMPAS/reweighting/reweighting_test',index=False)
    elif dataset == 'census':
        dataset_orig = AdultDataset(
            instance_weights_name='fnlwgt',
            features_to_drop=[] 
        )

        dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        
        RW = Reweighing(unprivileged_groups=unprivileged_groups,
                        privileged_groups=privileged_groups)
        dataset_transf_train = RW.fit_transform(dataset_orig_train)
        df,_ = dataset_transf_train.convert_to_dataframe(de_dummy_code=True,set_category=False)
        test_df,_=dataset_orig_test.convert_to_dataframe(de_dummy_code=True,set_category=False)
        df.to_csv('./data/Census/reweighting/reweighting_train',index=False)
        test_df.to_csv('./data/Census/reweighting/reweighting_test',index=False)
    elif dataset == 'credit':
        dataset_orig = GermanDataset(
            protected_attribute_names=['sex'],

        )

        dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        RW = Reweighing(unprivileged_groups=unprivileged_groups,
                        privileged_groups=privileged_groups)
        dataset_transf_train = RW.fit_transform(dataset_orig_train)
        df,_ = dataset_transf_train.convert_to_dataframe(de_dummy_code=True,set_category=False)
        test_df,_=dataset_orig_test.convert_to_dataframe(de_dummy_code=True,set_category=False)
        df.to_csv('./data/Credit/reweighting/reweighting_train',index=False)
        test_df.to_csv('./data/Credit/reweighting/reweighting_test',index=False)
    


def reweighing_evaluate(dataset='compas'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE=128    
    if dataset=='compas':
        # COMPAS reweighting
        compas_order=['sex','age_cat','race','juv_fel_count','juv_misd_count','juv_other_count','priors_count','days_b_screening_arrest','c_jail_time (days)','date_dif_in_jail','c_charge_degree','is_recid','decile_score','two_year_recid']
        train_df=pd.read_csv('data/COMPAS/reweighting/reweighting_train')
        test_df=pd.read_csv('data/COMPAS/reweighting/reweighting_test')
        df=pd.concat([train_df,test_df])
        df=df[compas_order]
        df_binary, Y, S, Y_true = transform_dataset_compas_reweight(df)
        Y = Y.to_numpy()
        print(np.mean(Y))

        l_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
        x_tensor = torch.tensor(df_binary.to_numpy().astype(np.float32))
        y_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
        s_tensor = torch.tensor(preprocessing.OneHotEncoder().fit_transform(np.array(S).reshape(-1, 1)).toarray())

        tv_dataset = TensorDataset(x_tensor[:len(train_df)+1], y_tensor[:len(train_df)+1], l_tensor[:len(train_df)+1], s_tensor[:len(train_df)+1])  # dataset = CustomDataset(x_tensor, y_tensor)
        print(x_tensor.shape[1])
        base_size = len(tv_dataset) // 10
        split = [7 * base_size,len(tv_dataset) - 7 * base_size]  # Train, validation, test
        train_dataset, val_dataset = random_split(tv_dataset, split)


        test_dataset = TensorDataset(x_tensor[len(train_df)+1:], y_tensor[len(train_df)+1:], l_tensor[len(train_df)+1:], s_tensor[len(train_df)+1:])  # dataset = CustomDataset(x_tensor, y_tensor)


        global_results = []
        for i in range(1):

            train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            # train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
            val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
            test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

            x_train_tensor = train_dataset[:][0]
            y_train_tensor = train_dataset[:][1]
            l_train_tensor = train_dataset[:][2]
            s_train_tensor = train_dataset[:][3]
            print(x_train_tensor.shape[1])


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
    elif dataset=='census':
        census_order = ['age','workclass','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income-per-year']
        train_df=pd.read_csv('data/Census/reweighting/reweighting_train')
        test_df=pd.read_csv('data/Census/reweighting/reweighting_test')
        df=pd.concat([train_df,test_df])
        df=df[census_order]
        df_binary, Y, S, Y_true = transform_dataset_census_reweight(df)
        print(np.mean(Y))

        l_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
        x_tensor = torch.tensor(df_binary.to_numpy().astype(np.float32))
        y_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
        s_tensor = torch.tensor(preprocessing.OneHotEncoder().fit_transform(np.array(S).reshape(-1, 1)).toarray())

        tv_dataset = TensorDataset(x_tensor[:len(train_df)+1], y_tensor[:len(train_df)+1], l_tensor[:len(train_df)+1], s_tensor[:len(train_df)+1])  # dataset = CustomDataset(x_tensor, y_tensor)
        print(x_tensor.shape[1])
        base_size = len(tv_dataset) // 10
        split = [7 * base_size,len(tv_dataset) - 7 * base_size]  # Train, validation, test
        train_dataset, val_dataset = random_split(tv_dataset, split)


        test_dataset = TensorDataset(x_tensor[len(train_df)+1:], y_tensor[len(train_df)+1:], l_tensor[len(train_df)+1:], s_tensor[len(train_df)+1:])  # dataset = CustomDataset(x_tensor, y_tensor)


        global_results = []
        for i in range(1):

            train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            # train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
            val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
            test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

            x_train_tensor = train_dataset[:][0]
            y_train_tensor = train_dataset[:][1]
            l_train_tensor = train_dataset[:][2]
            s_train_tensor = train_dataset[:][3]
            print(x_train_tensor.shape[1])


            # get the classification threshold, we use the same scale for compas so 4 instead of 0.5
            ori_start=time.time()
            threshold = 0.5

            net, results = train_and_evaluate(train_loader, val_loader, test_loader, device, input_shape=x_tensor.shape[1],
                                                grl_lambda=50,dataset='census')
            ori_end=time.time()
            ori_cost_time=ori_end-ori_start
            print('time costs:{} s'.format(ori_cost_time))

            result = get_metrics(results, threshold, 0)
            global_results.append(result)
    elif dataset == 'credit':
        credit_order = ['status', 'month', 'credit_history', 'purpose', 'credit_amount', 'savings', 'employment', 'investment_as_income_percentage', 'sex', 'other_debtors',
        'residence_since','property', 'age', 'installment_plans', 'housing', 'number_of_credits', 'skill_level', 'people_liable_for', 'telephone', 'foreign_worker', 'credit']
        df=pd.read_csv('data/Credit/reweighting/reweighting_train')
        df=df[credit_order]
        df_binary, Y, S, Y_true = transform_dataset_credit_reweight(df)
        print(np.mean(Y))

        l_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
        x_tensor = torch.tensor(df_binary.to_numpy().astype(np.float32))
        y_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
        s_tensor = torch.tensor(preprocessing.OneHotEncoder().fit_transform(np.array(S).reshape(-1, 1)).toarray())

        dataset = TensorDataset(x_tensor, y_tensor, l_tensor, s_tensor)  # dataset = CustomDataset(x_tensor, y_tensor)

        base_size = len(dataset) // 10
        split = [7 * base_size,len(dataset) - 7 * base_size]  # Train, validation, test
        train_dataset, val_dataset = random_split(dataset, split)

        df=pd.read_csv('data/Credit/reweighting/reweighting_test')
        df=df[credit_order]
        df_binary, Y, S, Y_true = transform_dataset_credit_reweight(df)
        print(np.mean(Y))

        l_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
        x_tensor = torch.tensor(df_binary.to_numpy().astype(np.float32))
        y_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
        s_tensor = torch.tensor(preprocessing.OneHotEncoder().fit_transform(np.array(S).reshape(-1, 1)).toarray())

        dataset = TensorDataset(x_tensor, y_tensor, l_tensor, s_tensor)  # dataset = CustomDataset(x_tensor, y_tensor)
        test_dataset = dataset

        global_results = []
        for i in range(1):

            train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            # train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
            val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
            test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

            x_train_tensor = train_dataset[:][0]
            y_train_tensor = train_dataset[:][1]
            l_train_tensor = train_dataset[:][2]
            s_train_tensor = train_dataset[:][3]



            # get the classification threshold, we use the same scale for compas so 4 instead of 0.5
            ori_start=time.time()
            threshold = 0.5

            net, results = train_and_evaluate(train_loader, val_loader, test_loader, device, input_shape=x_tensor.shape[1],
                                                grl_lambda=50,dataset='credit')
            ori_end=time.time()
            ori_cost_time=ori_end-ori_start
            print('time costs:{} s'.format(ori_cost_time))

            result = get_metrics(results, threshold, 0)
            global_results.append(result)