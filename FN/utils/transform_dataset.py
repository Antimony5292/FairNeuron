import numpy as np
import pandas as pd
from sklearn import preprocessing

def transform_dataset(df):
    """

    :param df:
    :return: Tuple of the transformed dataset and the labels Y and S
    """

    df_binary = df[(df["race"] == "Caucasian") | (df["race"] == "African-American")]

    del df_binary['c_jail_in']
    del df_binary['c_jail_out']

    ##separated class from the rests of the features
    # remove unnecessary dimensions from Y -> only the decile_score remains
    Y = df_binary['decile_score']
    del df_binary['decile_score']
    Y_true = df_binary['two_year_recid']
    del df_binary['two_year_recid']
    del df_binary['score_text']

    S = df_binary['race']
    #del df_binary['race']
    #del df_binary['is_recid']

    print(df_binary.shape)

    # set sparse to False to return dense matrix after transformation and keep all dimensions homogeneous
    encod = preprocessing.OneHotEncoder(sparse=False)

    data_to_encode = df_binary.to_numpy()
    feat_to_encode = data_to_encode[:, 0]
    # print(feat_to_encode)
    # transposition
    feat_to_encode = feat_to_encode.reshape(-1, 1)
    # print(feat_to_encode)
    encoded_feature = encod.fit_transform(feat_to_encode)

    df_binary_encoded = pd.DataFrame(encoded_feature)

    feat_to_encode = data_to_encode[:, 1]
    feat_to_encode = feat_to_encode.reshape(-1, 1)
    encoded_feature = encod.fit_transform(feat_to_encode)


    df_binary_encoded = pd.concat([df_binary_encoded, pd.DataFrame(encoded_feature)], axis=1)

    feat_to_encode = data_to_encode[:, 2] == "Caucasian"
    feat_to_encode = feat_to_encode.reshape(-1, 1)
    encoded_feature = encod.fit_transform(feat_to_encode)

    df_binary_encoded = pd.concat([df_binary_encoded, pd.DataFrame(encoded_feature)], axis=1)

    # feature [2] [3] [4] [5] [6] [7] [8] has to be put between 0 and 1

    for i in range(3, 10):
        encoded_feature = data_to_encode[:, i]
        ma = np.amax(encoded_feature)
        mi = np.amin(encoded_feature)
        encoded_feature = (encoded_feature - mi) / (ma - mi)
        df_binary_encoded = pd.concat([df_binary_encoded, pd.DataFrame(encoded_feature)], axis=1)

    feat_to_encode = data_to_encode[:, 10]
    feat_to_encode = feat_to_encode.reshape(-1, 1)
    encoded_feature = encod.fit_transform(feat_to_encode)

    df_binary_encoded = pd.concat([df_binary_encoded, pd.DataFrame(encoded_feature)], axis=1)

    feat_to_encode = data_to_encode[:, 11]
    feat_to_encode = feat_to_encode.reshape(-1, 1)
    encoded_feature = encod.fit_transform(feat_to_encode)

    df_binary_encoded = pd.concat([df_binary_encoded, pd.DataFrame(encoded_feature)], axis=1)

    return df_binary_encoded, Y, S, Y_true



def transform_dataset_census(df):
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
    del df_replace["fnlwgt"]

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


def transform_dataset_credit(df):

    label_encoder = preprocessing.LabelEncoder()
    oh_encoder = preprocessing.OneHotEncoder(sparse=False)

    Y = np.array(df.iloc[:,-1] == 2)

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

    # print(S)

    return df_binary_encoded, Y, S, Y_true
