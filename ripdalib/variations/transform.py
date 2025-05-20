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


def _get_first_nonzero(matrix:torch.Tensor):
    """
    Get the first non-zero element of a matrix, rows first (for line, for column in line).
    """
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j] > 0:
                return (i,j)


def middlepoints_to_onsets(corr_map:torch.Tensor, pattern:torch.Tensor):
    """
    Translate the correlation map such as the correlation coresponds to onsets of the pattern.
    
    Parameters
    ----------
    corr_map : torch.Tensor
        The correlation map to be translated.
    pattern : torch.Tensor
        The original pattern to be used for the translation.
    
    Returns
    -------
    torch.Tensor
        The translated correlation map.
    """

    if len(pattern.shape) != 2:
        raise NotImplementedError("Only 2D patterns are supported for now.")
    centers = list()
    for dim in pattern.shape:
        if dim %2 == 0:
            centers.append(dim//2 -1)
        else:
            centers.append((dim+1)//2 -1)

    centers = torch.tensor(centers)
    onset_point = _get_first_nonzero(pattern)
    # create a large enough matrix to do the translations
    translation = torch.zeros((corr_map.shape[0] + max(onset_point[0],centers[0]), corr_map.shape[1] + max(onset_point[1], centers[1])), dtype=corr_map.dtype)
    translation[onset_point[0]:onset_point[0]+corr_map.shape[0], onset_point[1]:onset_point[1]+corr_map.shape[1]] = corr_map

    return translation[centers[0]:centers[0]+corr_map.shape[0], centers[1]:centers[1]+corr_map.shape[1]] # + onset_point[0] - onset_point[0]...

    