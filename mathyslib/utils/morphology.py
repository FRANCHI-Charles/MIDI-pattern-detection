

def erosion(X:list, S:list) -> list:
    """
    Compute the erosion of X by structuring element S, with S centered at the origin.
    """
    # Get the first point of S to determine the translation vector
    s0, s1 = S[0]

    # Translate S to be centered at the origin
    S_centered = [(so - s0, sp - s1) for (so, sp) in S]

    eroded = list()

    X = set(X)

    for (o, p) in X:
        # Check if translated structuring element is included in X
        if all((o + so, p + sp) in X for (so, sp) in S_centered):
            eroded.append((o, p))

    return eroded


def dilation(X:list, S:list) -> list:
    """
    Compute the dilation of set X by structuring element S.
    """
    dilated = list()

    X = set(X)

    # Get the first point of S to determine the translation vector
    s0, s1 = S[0]

    # Translate S to be centered at the origin
    S_centered = [(so - s0, sp - s1) for (so, sp) in S]

    for (o, p) in X:
        for (so, sp) in S_centered:
            dilated.append((o + so, p + sp))

    return dilated


# opening (erosion then dilation)
def opening(X:list, S:list) -> list:
    """
    Compute the opening of set X by structuring element S.
    """
    return dilation(erosion(X, S), S)


# morphological closing of a set X (dilation then erosion)
def closing(X:list, S:list) -> list:
    """
    Compute the closing of set X by structuring element S.
    """
    return erosion(dilation(X, S), S)