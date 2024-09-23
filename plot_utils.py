import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import seaborn as sns


def softmax(mat):
    e = np.exp(mat - np.max(mat))
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else: # dim = 2
        return e / np.sum(e, axis=1, keepdims=True)


def conf_mat(Yp, Y, lims=None):
    if lims is None:
        min_val = np.min(Y).astype(int)
        max_val = np.max(Y).astype(int)
    else:
        min_val, max_val = lims
    print(min_val, max_val)
    confusion_matrix = np.zeros((max_val-min_val+1, max_val-min_val+1), dtype=int)
    for i in range(min_val,max_val+1):
        for j in range(min_val,max_val+1):
            confusion_matrix[i-1,j-1] = np.sum(np.logical_and(Y == i, Yp == j))

    bottom = np.sum(confusion_matrix, axis=0)
    bottom[bottom == 0] = 1
    confusion_matrix = confusion_matrix / bottom
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, bins=np.round(np.linspace(0, 512, 10)[1:]).astype(int), title='Confusion Matrix'):
    plt.figure(figsize=(8,8))
    plt.title(title)
    plt.imshow(confusion_matrix, cmap='viridis')
    plt.colorbar()
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.xticks(range(len(bins)), bins)
    plt.yticks(range(len(bins)), bins)
    for i in range(len(bins)):
        for j in range(len(bins)):
            plt.text(j, i, f'{confusion_matrix[i,j]:.2f}', ha='center', va='center', color='black')
    plt.show()


# prediction smoothing
def smooth_probs(Y_pred, Y_test, decay_factor=0.9):
    avg_probs = None
    Y_smoothed = []
    for i in tqdm(range(len(Y_test))):
        probs = Y_pred[i]
        if avg_probs is None:
            avg_probs = probs
        else:
            avg_probs = decay_factor * avg_probs + (1 - decay_factor) * probs
        Y_smoothed.append(avg_probs)
        if Y_test[i] == 0:
            avg_probs = None
    
    return np.array(Y_smoothed)
