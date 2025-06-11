import torch
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt

from ML.utils import my_dilatation, my_erosion, correlation

def show_morp_op():
    a = np.zeros((5,5), dtype=np.uint8)
    a[1:4,1:4] = [[0,1,1],
                [1,1,1],
                [0,1,0]]
    a = torch.tensor(a)
    s = torch.tensor([[0,1,0],
                    [0,1,1],
                    [0,0,0]], dtype=torch.uint8)
    
    s = torch.cat((s, torch.zeros(3,1)), 1)
    s = torch.cat((s, torch.zeros(1,4)), 0)

    correlation(a,s)
    fig, axes = plt.subplots(2,2)
    axes[0,0].imshow(a, cmap="grey", vmin=0, vmax=1.1)
    axes[0,1].imshow(s, cmap="grey", vmin=0, vmax=1.1)
    axes[1,0].imshow(my_erosion(a,s).squeeze((0,1)), cmap="grey", vmin=0, vmax=1.1)
    axes[1,1].imshow(my_dilatation(a,s).squeeze((0,1)), cmap="grey", vmin=0, vmax=1.1)

    plt.show()


def multibatch_morp_op():
    a = np.zeros((2,1,5,5), dtype=np.uint8)
    a[:,:,1:4,1:4] = [[0,1,1],
                [1,1,1],
                [0,1,0]]
    a = torch.tensor(a)
    s = torch.tensor([[[[0,1,0],
                    [0,1,1],
                    [0,0,0]]],
                    [[[0,1,0],
                    [0,1,0],
                    [0,0,0]]]], dtype=torch.uint8)

    print(my_erosion(a,s).shape)
    fig, axes = plt.subplots(2,3)
    axes[0,0].imshow(a[0,0], cmap="grey", vmin=0, vmax=1.1)
    axes[0,1].imshow(s[0,0], cmap="grey", vmin=0, vmax=1.1)
    axes[0,2].imshow(s[1,0], cmap="grey", vmin=0, vmax=1.1)
    axes[1,1].imshow(my_dilatation(a,s)[0,0], cmap="grey", vmin=0, vmax=1.1)
    axes[1,2].imshow(my_dilatation(a,s)[1,0], cmap="grey", vmin=0, vmax=1.1)

    plt.show()


def erosion_report():
    erodilaobject = image.imread('erodilaobject.png')
    erodilaobject = np.array(np.array(erodilaobject)[:,:,0] < 0.5, dtype=np.int8)

    structuring = image.imread('erodilaS.png')
    structuring = np.array(np.array(structuring)[:,:,0] < 0.5, dtype=np.int8)

    _, axes = plt.subplots(1,3)
    axes[0].imshow(1-erodilaobject, cmap="grey", vmin=0, vmax=1)
    axes[0].set_title("Object to erode")
    axes[1].imshow(1-structuring, cmap="grey", vmin=0, vmax=1)
    axes[1].set_title("Structuring element\n" + r"Zoom $\times$5")
    axes[2].imshow(~my_erosion(erodilaobject,structuring)[0,0], cmap="grey", vmin=0, vmax=1)
    axes[2].set_title("Eroded Object")

    plt.show()


def dilatation_report():
    erodilaobject = image.imread('erodilaobject.png')
    erodilaobject = np.array(np.array(erodilaobject)[:,:,0] < 0.5, dtype=np.int8)

    structuring = image.imread('erodilaS.png')
    structuring = np.array(np.array(structuring)[:,:,0] < 0.5, dtype=np.int8)

    _, axes = plt.subplots(1,3)
    axes[0].imshow(1-erodilaobject, cmap="grey", vmin=0, vmax=1)
    axes[0].set_title("Object to dilate")
    axes[1].imshow(1-structuring, cmap="grey", vmin=0, vmax=1)
    axes[1].set_title("Structuring element\n" + r"Zoom $\times$5")
    axes[2].imshow(~my_dilatation(erodilaobject,structuring)[0,0], cmap="grey", vmin=0, vmax=1)
    axes[2].set_title("Dilated Object")

    plt.show()

dilatation_report()
#erosion_report()
#show_morp_op()
#multibatch_morp_op()
