from Model_Capsnet import Model_Capsnet
from Model_ELM import Model_ELM
from Evaluation import evaluation


def Model_HAN(Train_Data, Train_Target, Test_Data, Test_Target, sol=None):
    if sol is None:
        sol = [5, 5, 5, 5]
    Eval1, pred1 = Model_ELM(Train_Data, Train_Target, Test_Data, Test_Target, sol[:2].astype('int'))
    Eval2, pred2 = Model_Capsnet(Train_Data, Train_Target, Test_Data, Test_Target, sol[2:])

    pred = (pred1 + pred2) / 2
    Eval = evaluation(pred, Test_Target)
    return Eval
