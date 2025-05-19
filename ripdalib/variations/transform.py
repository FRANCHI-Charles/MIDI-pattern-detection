import torch
from fractions import Fraction

def _get_mindiv(points:list):
    return max([point[0].denominator for point in points])


def list_to_matrix(points:list, mindiv:int=None):
    """
    Convert a list of points to a matrix.
    The points are in the form (onset, pitch).
    The matrix is a binary image in the form (time, pitch).
    """
    if mindiv is None:
        mindiv = _get_mindiv(points)
    onsets = [point[0] for point in points]
    pitches = [point[1] for point in points]
    min_time = min(onsets)
    min_pitch = min(pitches)
    max_time = max(onsets) + 1
    max_pitch = max(pitches) + 1
    matrix = torch.zeros((int((max_time-min_time)*mindiv), max_pitch-min_pitch), dtype=torch.int8)
    for point in points:
        matrix[round((point[0]-min_time)*mindiv), point[1]-min_pitch] = 1

    # Remove empty rows at the end
    while sum(matrix[-1]) == 0:
        matrix = matrix[:-1]
    return matrix

def matrix_to_list(matrix:torch.Tensor, mindiv:int=None):
    """
    Convert a matrix to a list of points.
    The matrix is a binary image in the form (time, pitch).
    The points are in the form (onset, pitch).
    """
    if mindiv is None:
        mindiv = 1
    points = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j] > 0:
                points.append((Fraction(i, mindiv), j))
    return points