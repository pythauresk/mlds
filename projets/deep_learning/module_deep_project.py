""" Custom functions for Deep Learning Project """
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np

def get_class_weight(initial_labels):
    """for training with class weight ponderation"""
    class_weight = Counter(initial_labels)
    total = len(initial_labels)
    for  key, value  in dict(class_weight).items():
        class_weight[key] = (1 / value)*(total)/2.0
    return class_weight

def split_dataset(features, labels, random_state):
    """Split into test, train and validation sets"""
    train_features, test_features, train_labels, test_labels = train_test_split(features,
                                                                            labels,
                                                                            test_size=0.2, 
                                                                            random_state=random_state)
    train_features, val_features, train_labels, val_labels = train_test_split(train_features,
                                                                          train_labels,
                                                                          test_size=0.2,
                                                                          random_state=random_state)
    return train_features, test_features, train_labels, test_labels, val_features, val_labels
    
# # # ----------- Plots -------------

def plot_roc(name, labels, predictions, **kwargs):
    """plot ROC curve https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#plot_the_roc"""
    fp, tp, _ = roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

def plot_metrics(history, metrics=['loss', 'auc', 'precision', 'recall']):
    """Modified version of https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#check_training_history"""
    if isinstance(metrics, str):
        metrics = [metrics]
    if len(metrics) > 4:
        raise ValueError('metrics argument must contain maximum 4 metrics')
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        if len(metrics) == 2:
            plt.subplot(1,2,n+1)
        elif len(metrics) >= 3:
            plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric],
                 color='blue', label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color='blue', linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
    plt.legend()

def plot_cm(labels, predictions, p=0.5, verbose=False):
    """Modified version of https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#evaluate_metrics"""
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    if verbose:
        print('True Negatives: ', cm[0][0])
        print('False Positives: ', cm[0][1])
        print('False Negatives: ', cm[1][0])
        print('True Positives: ', cm[1][1])
        
def plot_proportion(labels, resampled_labels, labels_name=['class 0','class 1']):
    """ Display proportion before and after resampling (over or under sampling)"""
    a = Counter(labels)
    b = Counter(resampled_labels)

    plt.subplot(1,2,1)
    plt.pie([a[0], a[1]],
            labels = labels_name,
            autopct = lambda x: str(round(x, 2)) + '%')
    plt.legend([a[0], a[1]], title='Nbr by class') 
    plt.title('Proportion before resampling')
    
    plt.subplot(1,2,2)
    plt.pie([b[0], b[1]], 
            labels = labels_name,
            autopct = lambda x: str(round(x, 2)) + '%')
    plt.legend([b[0], b[1]], title='Nbr by class') 
    plt.title('Proportion after resampling')
        