import torch
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
plt.style.use('fivethirtyeight')

def odds(prob):
    return prob / (1 - prob)

def log_odds(prob):
    return np.log(odds(prob))

def figure1(X_train, y_train, X_val, y_val, cm_bright=None):
    if cm_bright is None:
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)  # , edgecolors='k')
    ax[0].set_xlabel(r'$X_1$')
    ax[0].set_ylabel(r'$X_2$')
    ax[0].set_xlim([-2.3, 2.3])
    ax[0].set_ylim([-2.3, 2.3])
    ax[0].set_title('Generated Data - Train')

    ax[1].scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap=cm_bright)  # , edgecolors='k')
    ax[1].set_xlabel(r'$X_1$')
    ax[1].set_ylabel(r'$X_2$')
    ax[1].set_xlim([-2.3, 2.3])
    ax[1].set_ylim([-2.3, 2.3])
    ax[1].set_title('Generated Data - Validation')
    fig.tight_layout()
    plt.show()

    return fig


def figure2(prob1):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    prob = np.linspace(.01, .99, 99)

    for i in [0, 1]:
        ax[i].plot(prob, odds(prob), linewidth=2)
        ax[i].set_xlabel('Probability')
        if i:
            ax[i].set_yscale('log')
            ax[i].set_ylabel('Odds Ratio (log scale)')
            ax[i].set_title('Odds Ratio (log scale)')
        else:
            ax[i].set_ylabel('Odds Ratio')
            ax[i].set_title('Odds Ratio')
        ax[i].scatter([prob1, .5, (1 - prob1)], [odds(prob1), odds(.5), odds(1 - prob1)], c='r')
    plt.show();
    fig.tight_layout()

    return fig
def figure3(prob1):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    prob = np.linspace(.01, .99, 99)
    ax[0].plot(prob, log_odds(prob), linewidth=2)
    ax[0].set_xlabel('Probability')
    ax[0].set_ylabel('Log Odds Ratio')
    ax[0].set_title('Log Odds Ratio')
    ax[0].scatter([prob1, .5, (1 - prob1)], [log_odds(prob1), log_odds(.5), log_odds(1 - prob1)], c='r')

    ax[1].plot(log_odds(prob), prob, linewidth=2)
    ax[1].set_ylabel('Probability')
    ax[1].set_xlabel('Log Odds Ratio')
    ax[1].set_title('Probability')
    ax[1].scatter([log_odds(prob1), log_odds(.5), log_odds(1 - prob1)], [prob1, .5, (1 - prob1)], c='r')
    fig.tight_layout()
    plt.show()
    return fig
