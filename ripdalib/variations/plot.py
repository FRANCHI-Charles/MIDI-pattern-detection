import matplotlib.pyplot as plt
from torch import Tensor

def plot_matrix(matrix:Tensor):
    """
    Plot a binary matrix.
    The matrix is a binary image in the form (time, pitch).
    """
    matrix = matrix.cpu()
    plt.figure(figsize=(20, 10))
    plt.imshow(matrix.T, cmap='gray', origin="lower", interpolation='nearest')
    plt.show()

def plot_correlation(cor:Tensor):
    """
    Plot a correlation matrix.
    The matrix is in the form (time, pitch).
    """
    cor = cor.cpu()
    plt.figure(figsize=(20, 10))
    plt.imshow(cor.T, cmap='Reds', origin="lower", interpolation='nearest')
    # for (j,i),label in np.ndenumerate(cor):
    #     if label > 0:
    #         plt.text(i,j,label,ha='center',va='center')
    plt.show()