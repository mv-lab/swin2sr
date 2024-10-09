"""
Here, we filter and club together the predicted patches to form the predicted frame.
"""
import os

import numpy as np
from PIL import Image


class PredFrameCreator:

    def __init__(self, grid_index_manager, frame_t, dump_dir=None) -> None:
        self._grid_index_manager = grid_index_manager
        _, H, W, C = self._grid_index_manager.get_data_shape()
        self.frame = np.zeros((C, H, W), dtype=np.int32)
        self.target_frame = np.zeros((C, H, W), dtype=np.int32)
        self._frame_t = frame_t
        self._dump_dir = dump_dir
        os.makedirs(self._dump_dir, exist_ok=True)
        os.makedirs(self.ch_subdir(0), exist_ok=True)
        os.makedirs(self.ch_subdir(1), exist_ok=True)

        print(f'{self.__class__.__name__} frame_t:{self._frame_t}')

    def _update(self, predictions, indices, output_frame):
        for i, index in enumerate(indices):
            h, w, t = self._grid_index_manager.hwt_from_idx(index)
            if t != self._frame_t:
                continue
            sz = predictions[i].shape[-1]
            output_frame[:, h:h + sz, w:w + sz] = predictions[i]

    def update(self, predictions, indices):
        self._update(predictions, indices, self.frame)

    def update_target(self, target, indices):
        self._update(target, indices, self.target_frame)

    def reset(self):
        self.frame = np.zeros_like(self.frame)

    def dump_target(self):
        assert self._dump_dir is not None
        fname = os.path.join(self.ch_subdir(0), f"tar_t_{self._frame_t}.png")
        Image.fromarray(self.target_frame[0]).save(fname)
        fname = os.path.join(self.ch_subdir(1), f"tar_t_{self._frame_t}.png")
        Image.fromarray(self.target_frame[1]).save(fname)

    def ch_subdir(self, ch_idx):
        return os.path.join(self._dump_dir, f"ch_{ch_idx}")

    def dump(self, epoch):
        assert self._dump_dir is not None
        for ch_idx in range(self.frame.shape[0]):
            subdir = self.ch_subdir(ch_idx)
            fpath = os.path.join(subdir, f"{epoch}_t_{self._frame_t}.png")
            Image.fromarray(self.frame[ch_idx]).save(fpath)
