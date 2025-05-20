import matplotlib.pyplot as plt
from torch import Tensor
import numpy as np

from ripdalib.variations.transform import list_to_matrix, matrix_to_list, _get_mindiv
from ripdalib.utils.morphology import dilation, erosion

from ML.utils import my_dilatation


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


def plot_patterns_from_correlation(cor:Tensor, piece:list | Tensor, pattern:list | Tensor, threshold:float=0.5, quantization:int=None):
    """
    Plot the variated patterns from the correlation map with the corresponding threshold.
    The piece is in blue, the exact pattern is in green and the new patterns are in red.

    Parameters
    ----------
    cor : Tensor
        The correlation map.
    piece : list | Tensor
        A list of points (onset, pitch) or a binary matrix.
    pattern : list | Tensor
        A list of points (onset, pitch) or a binary matrix.
    threshold : float
        The threshold for the correlation map.
    quantization : int
        The quantization of the piece and pattern. If None, it will be calculated from the piece.
    """
    if quantization is None:
        quantization = _get_mindiv(piece)

    cor = cor.cpu()
    if isinstance(piece, Tensor):
        piece = piece.cpu()
        piece_matrix = piece
        piece = matrix_to_list(piece_matrix, quantization)
    else:
        piece_matrix = list_to_matrix(piece, quantization)

    if isinstance(pattern, Tensor):
        pattern = pattern.cpu()
        pattern_matrix = pattern
        pattern = matrix_to_list(pattern_matrix, quantization)
    else:
        pattern_matrix = list_to_matrix(pattern, quantization)

    

    new_patterns = my_dilatation(cor>=threshold, pattern_matrix).squeeze((0, 1)) * piece_matrix

    new_patterns_list = matrix_to_list(new_patterns, mindiv=quantization)

    translations = piece[np.argmin([piece[i][0] for i in range(len(piece))])]
    translations = [translations[0], translations[1] - new_patterns_list[0][1]]

    new_patterns_list = [(note[0] +translations[0], note[1] + translations[1]) for note in new_patterns_list]

    pattern_all = dilation(erosion(piece, pattern), pattern)

    plt.figure(figsize=(20, 10))
    plt.scatter(*zip(*piece))
    plt.scatter(*zip(*new_patterns_list), color='red')
    plt.scatter(*zip(*pattern_all), color='green')
    plt.show()