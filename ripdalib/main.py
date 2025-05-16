import numpy as np

from ripdalib.utils.morphology import opening, erosion


def ripda(points:list, Lmax:int=20, step:float=0.25, N:int=4, L_list:list=None) -> list:
    """
    TO COMPLETE.
  
    Parameters
    ----------
        points : list of tuples of dim 2
            List of points (Onset, Pitch)
        Lmax : int
            Maximum length for patterns.
        step : float
            Step size for L values.
        N : int
            Number of repetitions of the pattern.
        L_list : list of floats
            List of lengths to test. If None, will use np.arange(step, Lmax, step).
    """
    if L_list is not None:
        L_values = L_list
    else:
        L_values = np.arange(step, Lmax, step)
        
    results = []
    for L in L_values:
        O_L = [(i*L, 0) for i in range(N)] # structuring element for finding patterns that are repeated N times
        P = erosion(points, O_L) # find all points in the patterns P = erosion(X , O_L)

        if len(P) > 0:
            # print("L =", L)
            # print("P =", P)
            results.append((P, L))

    return results


def _windowing(P, L):
    """
    Splits the pattern P into multiple sub-patterns of exact length L.
    - Ensures each window starts at a valid onset.
    - Prevents empty windows.
    """
    if not P:
        return []

    P = sorted(P, key=lambda x: x[0])  # Sort by onset
    windowed_patterns = []
    i = 0  # Index to track starting point

    while i < len(P):
        t1 = P[i][0]  # Base onset for this window
        window = [p for p in P if t1 <= p[0] < t1 + L]

        if not window:  # Skip empty windows
            i += 1
            continue

        windowed_patterns.append(window)

        # Move to next onset after the current window
        while i < len(P) and P[i][0] < t1 + L:
            i += 1

    return windowed_patterns

def _filtering(P_list, P_full, L):
    if not P_list:
        return []

    P_to_test = [tuple(map(tuple, Pj)) for Pj in P_list]
    P_output = []
    P_all = P_to_test.copy()
    P_to_test = sorted(P_to_test, key=lambda pat: min(p[0] for p in pat))

    while P_to_test:
        Pj = P_to_test.pop(0)
        Pj_list = list(Pj)
        onset_j = min(p[0] for p in Pj_list)
        opening_Pj = opening(P_full, Pj_list)

        Nj = 4
        absorbed = []

        # This list tracks all valid successive window onsets relative to Pj
        reference_onsets = [onset_j]

        for Pk in P_all:
            if Pk == Pj or Pk not in P_to_test:
                continue

            Pk_list = list(Pk)
            onset_k = min(p[0] for p in Pk_list)

            if any(p in opening_Pj for p in Pk_list):
                # Check if onset_k is in a window immediately after one of the reference windows
                window_diff_ok = any(round((onset_k - ref_onset) / L) == 1 for ref_onset in reference_onsets)

                if window_diff_ok:
                    Nj += 1
                    if all(p in opening_Pj for p in Pk_list):
                        absorbed.append(Pk)
                        reference_onsets.append(onset_k)  # Extend the chain!

        for Pk in absorbed:
            if Pk in P_to_test:
                P_to_test.remove(Pk)

        P_output.append((Pj_list, Nj))

    return P_output


def ripda_bis(points:list, Lmax:int=20, step:float=0.25, N:int=4):
    """
    Extracts translation-invariant patterns from input list points for various L values.
    - Applies erosion to detect patterns.
    - Uses windowing to extract patterns of exact length L.
    - Uses filtering to refine extracted patterns.

    Parameters
    ----------
        points : list of tuples of dim 2
            List of points (Onset, Pitch)
        Lmax : int
            Maximum length for patterns.
        step : float
            Step size for L values.
        N : int
            Number of repetitions in the pattern.
    """
    found_patterns = []
    L_values = np.arange(step, Lmax, step)

    for L in L_values:
        # Step 1: Define structuring element O_L as a list
        O_L = [(i*L, 0) for i in range(N)]

        # Step 2: Apply erosion to find patterns
        P = erosion(points, O_L)

        # Step 3: Windowing to get patterns of exact length L
        P1_to_Pn = _windowing(P, L)
        # print(f"Windowed patterns (L={L}):", P1_to_Pn)

        # Step 4: Filtering to refine patterns
        filtered_patterns = _filtering(P1_to_Pn, P, L)
        # print(f"Filtered patterns (L={L}):", filtered_patterns)

        # Step 5: Store results
        for P_prime, N in filtered_patterns:
            found_patterns.append((P_prime, N, float(L)))

    return found_patterns