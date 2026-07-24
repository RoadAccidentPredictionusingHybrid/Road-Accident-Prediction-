import numpy as np
import os
import pandas as pd
from numpy import matlib
from sklearn import decomposition
from sklearn.manifold import TSNE
from scipy.stats import kurtosis, skew
from BFO import BFO
from CO import CO
from Global_vars import Global_vars
from MAO import MAO
from Model_Capsnet import Model_Capsnet
from Model_ELM import Model_ELM
from Model_HAN import Model_HAN
from Model_LSTM import Model_LSTM
from Obj_fun import objfun_feat, objfun_cls
from PROPOSED import PROPOSED
from Plot_Results import *
from TSO import TSO


def stats(val):
    v = np.zeros(9)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    v[5] = np.var(val)  # varience
    v[6] = kurtosis(val)  # kurtosis
    v[7] = skew(val)  # skewness
    v[8] = np.corrcoef(val)  # Correlation coefficient
    return v


# Read the dataset
an = 0
if an == 1:
    accidents = pd.read_csv('./Dataset/accidents.csv', index_col='Accident_Index')
    vehicles = pd.read_csv('./Dataset/vehicles.csv', on_bad_lines="skip", index_col='Accident_Index')
    casualties = pd.read_csv('./Dataset/casualties.csv', on_bad_lines="skip", index_col='Accident_Index')

    accidents.head()
    vehicles.head()
    casualties.head()
    accidents = accidents.join(vehicles, how='outer')

    # combining two columns
    accidents['Date_time'] = accidents['Date'] + ' ' + accidents['Time']

    for col in accidents.columns:
        accidents = (accidents[accidents[col] != -1])
    for col in casualties.columns:
        casualties = (casualties[casualties[col] != -1])

    accidents['Date_time'] = pd.to_datetime(accidents.Date_time)
    accidents.drop(['Date', 'Time'], axis=1, inplace=True)
    accidents.dropna(inplace=True)

    accident_ml = accidents.drop('Accident_Severity', axis=1)
    accident_ml = accident_ml[
        ['Did_Police_Officer_Attend_Scene_of_Accident', 'Age_of_Driver', 'Vehicle_Type', 'Age_of_Vehicle',
         'Engine_Capacity_(CC)', 'Day_of_Week', 'Weather_Conditions', 'Road_Surface_Conditions'
            , 'Light_Conditions', 'Sex_of_Driver', 'Speed_limit']]

    accident_ml.head()

    Data = accident_ml.values
    Tar = accidents['Accident_Severity'].values

    Datas = np.asarray(Data)
    uni = np.unique(np.asarray(Tar))
    Target = np.zeros((len(Tar), len(uni))).astype('int')
    for j in range(len(uni)):
        ind = np.where(np.asarray(Tar) == uni[j])
        Target[ind[0], j] = 1

    np.save('Data.npy', Datas)
    np.save('Target.npy', Target)

#  Feature (Statistical Feature)
an = 0
if an == 1:
    data = np.load('Data.npy', allow_pickle=True)  # Load the Dataset
    Feat = []
    for j in range(len(data)):
        print(j, len(data))
        statistical = stats(data[j])  # Statistical Feature
        Feat.append(np.asarray(statistical))
    np.save('Stat_Feat.npy', np.asarray(Feat))  # Save the Feat 1

# Feature (PCA FEATURE)
an = 0
if an == 1:
    data = np.load('Data.npy', allow_pickle=True)  # Load the Dataset
    pca = decomposition.PCA(n_components=5)
    pca_feature = pca.fit_transform(data)
    np.save('PCA_Feat.npy', pca_feature)  # Save the Feat 2

# t-distributed Stochastic Neighbor Embedding (Tsne) Feature
an = 0
if an == 1:
    data = np.load('Data.npy', allow_pickle=True)  # Load the Dataset
    tsne = TSNE(n_components=3)
    tsne_feature = tsne.fit_transform(data)
    np.save('TSNE_feature.npy', tsne_feature)  # Save the Feat 3

# optimization for Weighted Feature Extraction
an = 0
if an == 1:
    Feat_1 = np.load('Stat_Feat.npy', allow_pickle=True)  # Load the Feat 1
    Feat_2 = np.load('PCA_Feat.npy', allow_pickle=True)  # Load the Feat 2
    Feat_3 = np.load('TSNE_feature.npy', allow_pickle=True)  # Load the Feat 3
    Target = np.load('Target.npy', allow_pickle=True)
    Global_vars.Feat_1 = Feat_1
    Global_vars.Feat_2 = Feat_2
    Global_vars.Feat_3 = Feat_3
    Global_vars.Target = Target
    Npop = 10
    Chlen = Feat_1.shape[1] + Feat_2.shape[1] + Feat_3.shape[1]  # No of Features from Extracted Features
    xmin = matlib.repmat(np.asarray([0.01]), Npop, Chlen)
    xmax = matlib.repmat(np.asarray([0.99]), Npop, Chlen)
    fname = objfun_feat
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("MAO...")
    [bestfit1, fitness1, bestsol1, time1] = MAO(initsol, fname, xmin, xmax, Max_iter)  # MAO

    print("TSO...")
    [bestfit2, fitness2, bestsol2, time2] = TSO(initsol, fname, xmin, xmax, Max_iter)  # TSO

    print("CO...")
    [bestfit3, fitness3, bestsol3, time3] = CO(initsol, fname, xmin, xmax, Max_iter)  # CO

    print("BFO...")
    [bestfit4, fitness4, bestsol4, time4] = BFO(initsol, fname, xmin, xmax, Max_iter)  # BFO

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # Enhanced BFO

    BestSol = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]

    np.save('BestSol_Feat.npy', np.asarray(BestSol))  # Bestsol classification

# Weighted Feature Fusion
an = 0
if an == 1:
    bests = np.load('BestSol_Feat.npy', allow_pickle=True)  # Load the Best sol
    Stat_Feat = np.load('Stat_Feat.npy', allow_pickle=True)  # Load the Feat 1
    PCA_Feat = np.load('PCA_Feat.npy', allow_pickle=True)  # Load the Feat 2
    TSNE_feature = np.load('TSNE_feature.npy', allow_pickle=True)  # Load the Feat 3
    Feat = np.concatenate((TSNE_feature, np.concatenate((Stat_Feat, PCA_Feat), axis=1)), axis=1)
    feature = []
    for j in range(Feat.shape[0]):
        Feature = Feat[j] * bests[4, :]
        feature.append(Feature)
    np.save('Weighted_Feature.npy', feature)  # Save the Feature

# optimization for Classification
an = 0
if an == 1:
    Feat = np.load('Weighted_Feature.npy', allow_pickle=True)  # Load the Images
    Target = np.load('Target.npy', allow_pickle=True)  # Load the Target
    Global_vars.Feat = Feat
    Global_vars.Target = Target
    Npop = 10
    Chlen = 4  # Hidden Neuron Count, Step per epoch in ELM, Hidden Neuron Count, Step per epoch in Capsnet
    xmin = matlib.repmat(np.asarray([5, 5, 5, 5]), Npop, 1)
    xmax = matlib.repmat(np.asarray([255, 50, 255, 50]), Npop, 1)
    fname = objfun_cls
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("MAO...")
    [bestfit1, fitness1, bestsol1, time1] = MAO(initsol, fname, xmin, xmax, Max_iter)  # (MAO)

    print("TSO...")
    [bestfit2, fitness2, bestsol2, time2] = TSO(initsol, fname, xmin, xmax, Max_iter)  # (TSO)

    print("CO...")
    [bestfit3, fitness3, bestsol3, time3] = CO(initsol, fname, xmin, xmax, Max_iter)  # (CO)

    print("BFO...")
    [bestfit4, fitness4, bestsol4, time4] = BFO(initsol, fname, xmin, xmax, Max_iter)  # BFO

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)  # PROPOSED

    BestSol = [bestsol1.squeeze(), bestsol2.squeeze(), bestsol3.squeeze(), bestsol4.squeeze(), bestsol5.squeeze()]
    fitness = [fitness1.squeeze(), fitness2.squeeze(), fitness3.squeeze(), fitness4.squeeze(), fitness5.squeeze()]

    np.save('Fitness.npy', np.asarray(fitness))
    np.save('BestSol_CLS.npy', np.asarray(BestSol))  # Bestsol classification

# KFOLD - Classification
an = 0
if an == 1:
    Feature = np.load('Weighted_Feature.npy', allow_pickle=True)
    BestSol = np.load('BestSol_CLS.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    K = 5
    Per = 1 / 5
    Perc = round(Feature.shape[0] * Per)
    EVAL = []
    for i in range(K):
        Eval = np.zeros((10, 25))
        Feat = Feature
        Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
        Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
        test_index = np.arange(i * Perc, ((i + 1) * Perc))
        total_index = np.arange(Feat.shape[0])
        train_index = np.setdiff1d(total_index, test_index)
        Train_Data = Feat[train_index, :]
        Train_Target = Target[train_index, :]
        for j in range(BestSol.shape[0]):
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Eval[j, :], pred = Model_HAN(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval[5, :], pred_1 = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[6, :], pred_2 = Model_Capsnet(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[7, :], pred_3 = Model_ELM(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[8, :], pred_4 = Model_HAN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[9, :] = Eval[4, :]
        EVAL.append(Eval)
    np.save('Eval_ALL_Fold.npy', np.asarray(EVAL))


plot_conv()
ROC_curve()
Plot_FKOLD()
Plot_Classes()
Plot_Epoch()
plot_Abliation()
