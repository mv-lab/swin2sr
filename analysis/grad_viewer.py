"""
This module computes the gradients and stores them so that next access is fast.
This can be used to compute gradients of arbitrary order on images. 
Last two dimensions of the data are assumed to be x & y dimension.

grads = GradientFetcher(imgs)
To get d/dx2y3,
grad_x2_y3 = grads[2,3]

"""
import numpy as np
from typing import List, Tuple
import seaborn as sns


class GradientFetcher:
    def __init__(self, data) -> None:
        self._data = data

        self._grad_data = {0: {0: self._data}}

    @staticmethod
    def apply_x_grad(data):
        grad = np.empty(data.shape)
        grad[:] = np.nan
        grad[..., :, 1:] = data[..., :, 1:] - data[..., :, :-1]
        return grad

    @staticmethod
    def apply_y_grad(data):
        grad = np.empty(data.shape)
        grad[:] = np.nan
        grad[..., 1:, :] = data[..., 1:, :] - data[..., :-1, :]
        return grad

    def __getitem__(self, order):
        order_x, order_y = order
        if order_x in self._grad_data and order_y in self._grad_data[order_x]:
            return self._grad_data[order_x][order_y]

        self.compute(order_x, order_y)
        return self._grad_data[order_x][order_y]

    def compute(self, order_x, order_y):
        assert order_y >= 0 and order_x >= 0
        if order_x in self._grad_data:
            if order_y in self._grad_data[order_x]:
                return self._grad_data[order_x][order_y]
            if order_y - 1 not in self._grad_data[order_x]:
                self.compute(order_x, order_y - 1)

            self._grad_data[order_x][order_y] = self.apply_y_grad(self._grad_data[order_x][order_y - 1])
            return self._grad_data[order_x][order_y]

        self._grad_data[order_x] = {}
        self.compute(order_x - 1, order_y)
        self._grad_data[order_x][order_y] = self.apply_x_grad(self._grad_data[order_x - 1][order_y])
        return self._grad_data[order_x][order_y]


class GradientViewer:
    def __init__(self, data) -> None:
        self._data = data
        self._grad = GradientFetcher(data)

    def plot(self,
             ax,
             gradorder_list: List[Tuple[int, int]],
             x_start=0,
             x_end=None,
             y_start=0,
             y_end=None,
             subsample=1,
             reduce_x=False,
             reduce_y=False):
        if x_end is None:
            x_end = self._data.shape[-1]

        if y_end is None:
            y_end = self._data.shape[-2]

        if isinstance(reduce_x, bool):
            reduce_x = [reduce_x] * len(gradorder_list)
        if isinstance(reduce_y, bool):
            reduce_y = [reduce_y] * len(gradorder_list)

        all_plots_data = []
        for idx, order in enumerate(gradorder_list):
            grad_data = self._grad[order]
            grad_data = grad_data[y_start:y_end:subsample, x_start:x_end:subsample]
            if reduce_x[idx]:
                grad_data = grad_data.mean(axis=1)
                sns.lineplot(data=grad_data, ax=ax[idx])
                all_plots_data.append(grad_data)
            elif reduce_y[idx]:
                grad_data = grad_data.mean(axis=0)
                sns.lineplot(data=grad_data, ax=ax[idx])
                all_plots_data.append(grad_data)
            else:
                sns.heatmap(grad_data, ax=ax[idx])
                all_plots_data.append(grad_data)
        return all_plots_data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    imgs = np.arange(1024).reshape(1, 1, 32, 32)
    plt.imshow(imgs[0, 0])
    grads = GradientFetcher(imgs)
    gradx = grads[1, 0]
    print('next')
    grady = grads[0, 1]
    print('next')
    gradxy = grads[1, 1]
