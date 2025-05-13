from collections import defaultdict

def _compute_vector_table(X):
    vectors = []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            vector = (X[j][0] - X[i][0], X[j][1] - X[i][1])
            vectors.append((vector, i))  # (vector, index of origin)
    return vectors

def SIA(X):
    """
    Applies the SIA (Structure Induction Algorithm) to a 2D point set X.
    Returns a list of maximal translatable patterns (MTPs), each as a tuple:
    (pattern, vector), where:
      - pattern is a list of points
      - vector is the common translation
    """
    # Step 1 – Sorting the dataset
    X = sorted(X)

    # Step 2 – Computing the vector table
    V = _compute_vector_table(X)

    # Step 3 – Sorting the vectors
    V_sorted = sorted(V, key=lambda vi: (vi[0], vi[1]))

    # Step 4 – Printing out (i.e., building the structure of SD(X))
    mtp_dict = defaultdict(list)
    for vector, i in V_sorted:
        mtp_dict[vector].append(i)

    SD = []

    for vector, indices in mtp_dict.items():
        if len(indices) >= 2:
            # Build the pattern = each origin point + the vector
            pattern = [X[i] for i in indices]
            # Append as (pattern, translation_vector)
            SD.append((pattern, vector))

    return SD


def _translate_pattern(P, v):
    """Translates pattern P by vector v."""
    return [(p[0] + v[0], p[1] + v[1]) for p in P]

def _compute_translators(P, D_set):
    """
    Computes T(P, D): the set of vectors v such that t(P, v) ⊆ D
    """
    translators = []
    for d in D_set:
        v = (d[0] - P[0][0], d[1] - P[0][1])
        translated_P = _translate_pattern(P, v)
        if all(pt in D_set for pt in translated_P):
            translators.append(v)
    return translators

def SIATEC(X):
    """
    Implements the SIATEC algorithm (David Meredith, 2002).
    - Finds all translational equivalence classes (TECs) in X.
    - Returns a list of tuples (pattern, translations, vector).
    """
    # Step 1 - Sorting the dataset
    X = sorted(X)
    D_set = set(X)
    n = len(X)

    # Step 2 – Computing W
    W = [(X[i], X[j]) for i in range(n) for j in range(i + 1, n)]

    # Step 3 – Computing V
    V = [((p_j[0] - p_i[0], p_j[1] - p_i[1]), i) for i, (p_i, p_j) in enumerate(W)]

    # Step 4 – Sorting V to produce V (same as step 3 in SIA)
    V_sorted = sorted(V, key=lambda vi: (vi[0], vi[1]))

    # Step 5 – “Vectorizing” the MTPs
    C = defaultdict(list)
    for vector, origin_idx in V_sorted:
        p_i = W[origin_idx][0]
        C[vector].append(p_i)

    # Step 6 – Sorting X
    X = sorted(X)

    # Step 7 – Printing out
    tecs = []
    seen_patterns = set()
    for vector, group in C.items():
        if len(group) < 2:
            continue
        P = tuple(sorted(group))  # canonical pattern
        if P in seen_patterns:
            continue
        seen_patterns.add(P)
        translators = _compute_translators(P, D_set)
        tecs.append((list(P), translators))

    return tecs