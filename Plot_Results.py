import numpy as np
import warnings
from matplotlib import pylab
from sklearn.metrics import roc_curve
from itertools import cycle
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import matplotlib
warnings.filterwarnings("ignore")

# https://stackoverflow.com/questions/5812960/change-figure-window-title-in-pylab

def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_conv():
    # matplotlib.use('TkAgg')
    matplotlib.use('Qt5Agg')

    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('Convergence')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'MAO-HAN', 'TSO-HAN', 'CO-HAN', 'BFO-HAN', 'MF-BFO-HAN']
    Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
    Conv_Graph = np.zeros((5, 5))
    for j in range(len(Algorithm) - 1):
        Conv_Graph[j, :] = stats(Fitness[j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Report Dataset 1',
          ' --------------------------------------------------')
    print(Table)
    length = np.arange(Fitness.shape[-1])
    Conv_Graph = Fitness

    plt.plot(length, Conv_Graph[0, :], color='#f97306', linewidth=3, label='MAO-HAN')
    plt.plot(length, Conv_Graph[1, :], color='#3d7afd', linewidth=3, label='TSO-HAN')
    plt.plot(length, Conv_Graph[2, :], color='#b9ff66', linewidth=3, label='CO-HAN')
    plt.plot(length, Conv_Graph[3, :], color='#bb3f3f', linewidth=3, label='BFO-HAN')
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, label='MF-BFO-HAN')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)

    plt.savefig("./Results/Convergence.png")
    plt.show()


def ROC_curve():
    lw = 2
    fig = pylab.gcf()
    fig.canvas.manager.set_window_title('ROC Curve')
    cls = ['LSTM', 'ELM', 'CapsNet', 'HAN', 'MF-BFO-HAN']
    Actual = np.load('Target.npy', allow_pickle=True).astype('int')
    colors = cycle(["#fe2f4a", "#0165fc", "#ffff14", "lime", "black"])
    for i, color in zip(range(len(cls)), colors):
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i],
        )
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    path = "./Results/ROC.png"
    plt.savefig(path)
    plt.show()


def Plot_FKOLD():
    eval = np.load('Eval_ALL_Fold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'FOR', 'NPV', 'FDR', 'F1_score',
             'MCC',
             'pt',
             'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    # Table_Term = [0, 2, 3, 5]
    Table_Term = [1, 9, 12]
    FOLD = [1, 2, 3, 4, 5]
    Algorithm = ['TERMS', 'MAO-HAN', 'TSO-HAN', 'CO-HAN', 'BFO-HAN', 'MF-BFO-HAN']
    Classifier = ['TERMS', 'LSTM', 'ELM', 'CapsNet', 'HAN', 'MF-BFO-HAN']
    for k in range(eval.shape[0]):
        value = eval[k, :, 4:]
        Table = PrettyTable()
        Table.add_column(Algorithm[0], (np.asarray(Terms))[np.asarray(Table_Term)])
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, Table_Term])
        print('-------------------------------------------------- ', str(FOLD[k]), ' Fold ',
              'Algorithm Comparison of Dataset', 0 + 1,
              '--------------------------------------------------')
        print(Table)
        Table = PrettyTable()
        Table.add_column(Classifier[0], (np.asarray(Terms))[np.asarray(Table_Term)])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, Table_Term])
        print('-------------------------------------------------- ', str(FOLD[k]), ' Fold ',
              'Classifier Comparison of Dataset', 0 + 1,
              '--------------------------------------------------')
        print(Table)



def Plot_Classes():
    eval = np.load('Eval_ALL_Classes.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'FOR', 'NPV', 'FDR', 'F1_score',
             'MCC',
             'pt',
             'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']

    Graph_Term = [0, 12, 16]
    for j in range(len(Graph_Term)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = eval[k, l, Graph_Term[j] + 4]
        length = np.arange(3)
        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])

        ax.plot(length, Graph[:, 0], color='#8cfd7e', linewidth=3, marker='D', markerfacecolor='red',
                markersize=6, label='MAO-HAN')
        ax.plot(length, Graph[:, 1], color='#6140ef', linewidth=3, marker='s', markerfacecolor='green',
                markersize=6, label='TSO-HAN')
        ax.plot(length, Graph[:, 2], color='#fe420f', linewidth=3, marker='H', markerfacecolor='cyan',
                markersize=8, label='CO-HAN')
        ax.plot(length, Graph[:, 3], color='#00ffff', linewidth=3, marker='p', markerfacecolor='#fdff38',
                markersize=8, label='BFO-HAN')
        ax.plot(length, Graph[:, 4], color='k', linewidth=3, marker='*', markerfacecolor='w', markersize=12,
                label='MF-BFO-HAN')
        plt.xticks(length, ('Class 1', 'Class 2', 'Class 3'))
        # plt.xlabel('Road Accident Severity', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
        path = "./Results/Classes_%s_lrean.png" % (Terms[Graph_Term[j]])

        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Road Accident Severity vs '+Terms[Graph_Term[j]])

        plt.savefig(path)
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.15, 0.1, 0.7, 0.8])
        X = np.arange(3)

        ax.bar(X + 0.00, Graph[:, 5], color='#0165fc', edgecolor='w', width=0.15, label="LSTM")
        ax.bar(X + 0.15, Graph[:, 6], color='#ff474c', edgecolor='w', width=0.15, label="ELM")
        ax.bar(X + 0.30, Graph[:, 7], color='#be03fd', edgecolor='w', width=0.15, label="CapsNet")
        ax.bar(X + 0.45, Graph[:, 8], color='#21fc0d', edgecolor='w', width=0.15, label="HAN")
        ax.bar(X + 0.60, Graph[:, 4], color='k', edgecolor='w', width=0.15, label="MF-BFO-HAN")
        plt.xticks(X + 0.15, ('Class 1', 'Class 2', 'Class 3'))
        # plt.xlabel('Road Accident Severity', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.ylabel(Terms[Graph_Term[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
        path = "./Results/Classes_%s_bar.png" % (Terms[Graph_Term[j]])

        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Road Accident Severity vs '+Terms[Graph_Term[j]])

        plt.savefig(path)
        plt.show()


def Plot_Epoch():
    eval = np.load('Eval_ALL_EPOCH.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'FOR', 'NPV', 'FDR', 'F1_score',
             'MCC',
             'pt',
             'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
    Graph_Terms = [0, 14, 16]
    for j in range(len(Graph_Terms)):
        Graph = np.zeros((eval.shape[0], eval.shape[1]))
        for k in range(eval.shape[0]):
            for l in range(eval.shape[1]):
                Graph[k, l] = eval[k, l, Graph_Terms[j] + 4]

        X = np.arange(6)

        plt.plot(X, Graph[:, 0], color='#f97306', linewidth=4, marker='o', markerfacecolor='blue', markersize=8,
                 label="MAO-HAN")
        plt.plot(X, Graph[:, 1], color='#0cff0c', linewidth=4, marker='o', markerfacecolor='red', markersize=8,
                 label="TSO-HAN")
        plt.plot(X, Graph[:, 2], color='#0504aa', linewidth=4, marker='o', markerfacecolor='green', markersize=8,
                 label="CO-HAN")
        plt.plot(X, Graph[:, 3], color='#ffa756', linewidth=4, marker='o', markerfacecolor='yellow', markersize=8,
                 label="BFO-HAN")
        plt.plot(X, Graph[:, 4], color='black', linewidth=4, marker='o', markerfacecolor='cyan', markersize=8,
                 label="MF-BFO-HAN")
        plt.xticks(X, ('50', '100', '150', '200', '250', '300'))
        plt.xlabel('EPOCH', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
        path = "./Results/Epoch_%s_line.png" % (Terms[Graph_Terms[j]])
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Road Accident Severity vs ' + Terms[Graph_Terms[j]])
        plt.savefig(path)
        plt.show()

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        X = np.arange(6)
        ax.bar(X + 0.00, Graph[:, 5], color='#ad8150', width=0.10, label="LSTM")
        ax.bar(X + 0.10, Graph[:, 6], color='#75bbfd', width=0.10, label="ELM")
        ax.bar(X + 0.20, Graph[:, 7], color='#e50000', width=0.10, label="CapsNet")
        ax.bar(X + 0.30, Graph[:, 8], color='#bf77f6', width=0.10, label="HAN")
        ax.bar(X + 0.40, Graph[:, 4], color='k', width=0.10, label="MF-BFO-HAN")
        plt.xticks(X + 0.10, ('50', '100', '150', '200', '250', '300'))
        plt.xlabel('EPOCH', fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.ylabel(Terms[Graph_Terms[j]], fontname="Arial", fontsize=12, fontweight='bold', color='#35530a')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=3, fancybox=True, shadow=False)
        path = "./Results/Epoch_%s_bar.png" % (Terms[Graph_Terms[j]])
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Road Accident Severity vs ' + Terms[Graph_Terms[j]])
        plt.savefig(path)
        plt.show()


def plot_Abliation():
    eval = np.load('Eval_all_ABLIATION.npy', allow_pickle=True)
    Terms = ['Accuracy']
    Classifier = ['TERMS', 'LSTM', 'BILSTM', 'Res_LSTM', 'LSTM_TCN', 'Capsnet', 'ELM', 'Bi_LSTM_Capsnet', 'Bi_LSTM_TCN',
                  'HAN', 'MF-BFO-HAN']
    Value = eval[0, 4, :, :]
    Table = PrettyTable()
    Table.add_column(Classifier[0], Terms)
    for j in range(len(Classifier) - 1):
        formatted_values = [f"{value:.2f}" for value in Value[j, :]]
        Table.add_column(Classifier[j + 1], formatted_values)
    print('-------------------------------------------------- Abliation Experiment',
          '--------------------------------------------------')
    print(Table)


if __name__ == '__main__':
    # plot_conv()
    # ROC_curve()
    Plot_FKOLD()
    # Plot_Classes()
    # Plot_Epoch()
    # plot_Abliation()
