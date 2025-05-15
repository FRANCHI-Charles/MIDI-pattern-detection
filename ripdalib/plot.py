import numpy as np
import matplotlib.pyplot as plt
from ripdalib.utils.morphology import dilation


def plot_ripda(points, found_patterns, N=4):
    """
    Affiche pour chaque pattern trouvé par algo_1 deux sous-graphes :
    - à gauche : le pattern trouvé dans les points originaux
    - à droite : la dilation du pattern par le structurant O_L = {0, L, 2L, 3L}
    """
    if not found_patterns:
        print("Aucun pattern trouvé.")
        return

    x_all, y_all = zip(*points)

    for idx, (pattern, L) in enumerate(found_patterns, start=1):
        if not pattern:
            continue

        color = np.random.rand(3,)
        x_p, y_p = zip(*pattern)

        # Structuring element O_L for N=4
        O_L = [(n * L, 0) for n in range(N)]
        dilated = dilation(pattern, O_L)
        x_d, y_d = zip(*dilated) if dilated else ([], [])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        # Left: pattern in original points
        axes[0].scatter(x_all, y_all, color='lightgrey', alpha=0.4, label='Points originaux')
        axes[0].scatter(x_p, y_p, color=color, label='Pattern trouvé')
        axes[0].set_title(f'Pattern #{idx} – L = {L:.2f}')
        axes[0].set_xlabel("Onset")
        axes[0].set_ylabel("Pitch")
        axes[0].grid(True)
        axes[0].legend()

        # Right: dilation
        axes[1].scatter(x_all, y_all, color='lightgrey', alpha=0.4, label='Points originaux')
        axes[1].scatter(x_d, y_d, color=color, label='Dilation')
        axes[1].set_title(f'Dilatation du pattern #{idx}')
        axes[1].set_xlabel("Onset")
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        plt.show()


def plot_ripda_bis(points, found_patterns):
    """
    Affiche pour chaque pattern deux sous-graphes :
    - à gauche : le pattern trouvé dans les points originaux
    - à droite : la dilation du pattern par son structurant O_L (défini avec le bon N)
    """
    x_all, y_all = zip(*points)

    for idx, (pattern, N, L) in enumerate(found_patterns, start=1):
        if not pattern:
            continue

        color = np.random.rand(3,)
        x_p, y_p = zip(*pattern)

        # Use N to define O_L properly
        O_L = [(n * L, 0) for n in range(N)]
        dilated = dilation(pattern, O_L)
        x_d, y_d = zip(*dilated) if dilated else ([], [])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

        # Left: original points + pattern
        axes[0].scatter(x_all, y_all, color='lightgrey', alpha=0.4, label='Points originaux')
        axes[0].scatter(x_p, y_p, color=color, label='Pattern trouvé')
        axes[0].set_title(f'Pattern #{idx} – L = {L:.2f}, N = {N}')
        axes[0].set_xlabel("Onset")
        axes[0].set_ylabel("Pitch")
        axes[0].grid(True)
        axes[0].legend()

        # Right: dilation with proper O_L
        axes[1].scatter(x_all, y_all, color='lightgrey', alpha=0.4, label='Points originaux')
        axes[1].scatter(x_d, y_d, color=color, label='Dilatation (N={})'.format(N))
        axes[1].set_title(f'Dilation du pattern #{idx}')
        axes[1].set_xlabel("Onset")
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        plt.show()