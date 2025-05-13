from typing import Tuple, Any

import torch

from deep_morpho.datasets.mnist_dataset import MnistMorphoDataset, ROOT_MNIST_DIR
from deep_morpho.datasets.mnist_base_dataset import resize_image


class MnistMorphoDatasetRectify(MnistMorphoDataset):

    def __init__(
        self,
        morp_operation,
        n_inputs: int = "all",
        threshold: float = 30,
        size=(50, 50),
        first_idx: int = 0,
        indexes=None,
        preprocessing=None,
        root: str = ROOT_MNIST_DIR,
        train: bool = True,
        invert_input_proba: bool = 0,
        do_symetric_output: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(morp_operation, n_inputs, threshold, size, first_idx, indexes, preprocessing, root, train, invert_input_proba, do_symetric_output, **kwargs)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        input_ = (resize_image(self.data[index].numpy(), self.size) >= (self.threshold))[None, ...] # Permutation HERE

        if torch.rand(1) < self.invert_input_proba:
            input_ = 1 - input_

        target = self.morp_operation(input_).squeeze(0).float()
        input_ = torch.tensor(input_).float()
        # input_ = input_.permute(1, 2, 0)  # From numpy format (W, L, H) to torch format (L, H, W) CHANGES HERE
        # target = target.permute(1, 2, 0)  # From numpy format (W, L, H) to torch format (L, H, W)

        if self.preprocessing is not None:
            input_ = self.preprocessing(input_)
            target = self.preprocessing(target)

        if self.do_symetric_output:
            return 2 * input_ - 1, 2 * target - 1

        return input_.float(), target.float()
