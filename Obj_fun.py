import numpy as np
from Evaluation import evaluation
from Global_vars import Global_vars
from Model_HAN import Model_HAN
from RELIEF import reliefF


def objfun_feat(Soln):
    Feat_1 = Global_vars.Feat_1
    Feat_2 = Global_vars.Feat_2
    Feat_3 = Global_vars.Feat_3
    Tar = Global_vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            FEAT = []
            Feat = np.concatenate((Feat_3, np.concatenate((Feat_1, Feat_2), axis=1)), axis=1)
            for j in range(Feat.shape[0]):
                sol = Soln[i, :]
                Feature = Feat[j, :] * sol
                FEAT.append(Feature)
            score = reliefF(np.asarray(FEAT), np.ravel(Tar))
            Fitn[i] = 1 / score
        return Fitn
    else:
        sol = Soln
        FEAT = []
        Feat = np.concatenate((Feat_3, np.concatenate((Feat_1, Feat_2), axis=1)), axis=1)
        for j in range(Feat.shape[0]):
            Feature = Feat[j, :] * sol
            FEAT.append(Feature)
        score = reliefF(np.asarray(FEAT), np.ravel(Tar))
        Fitn = 1 / score
        return Fitn


def objfun_cls(Soln):
    Feat = Global_vars.Feat
    Tar = Global_vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Feat.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = Feat[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = Feat[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, pred = Model_HAN(Train_Data, Train_Target, Test_Data, Test_Target, sol=sol)
            Eval = evaluation(pred, Test_Target)
            Fitn[i] = (1 / Eval[4] + Eval[7])  # Accuracy + Precision

        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Feat[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = Feat[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, pred = Model_HAN(Train_Data, Train_Target, Test_Data, Test_Target, sol=sol)
        Eval = evaluation(pred, Test_Target)
        Fitn = (1 / Eval[4] + Eval[7])  # Accuracy + Precision
        return Fitn
