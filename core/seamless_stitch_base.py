"""
SeamlessStitchBase class will ensure the basic functionality
"""
import torch


class SeamlessStitchBase:
    def __init__(self, grid_size, stitched_frame):
        assert len(stitched_frame.shape) == 4, 'Frame should be of shape (num_images,H,W,2)'
        self._data = stitched_frame
        self._sz = grid_size
        self._N = stitched_frame.shape[-1] // self._sz
        assert stitched_frame.shape[-1] % self._sz == 0

    def patch_location(self, row_idx, col_idx):
        """
        Top left location of the patch
        """
        return self._sz * row_idx, self._sz * col_idx

    def get_lboundary(self, row_idx, col_idx):
        h, w = self.patch_location(row_idx, col_idx)
        return self._data[..., h:h + self._sz, w:w + 1]

    def get_rboundary(self, row_idx, col_idx):
        h, w = self.patch_location(row_idx, col_idx)
        return self._data[..., h:h + self._sz, w + self._sz - 1:w + self._sz]

    def get_tboundary(self, row_idx, col_idx):
        h, w = self.patch_location(row_idx, col_idx)
        return self._data[..., h:h + 1, w:w + self._sz]

    def get_bboundary(self, row_idx, col_idx):
        h, w = self.patch_location(row_idx, col_idx)
        return self._data[..., h + self._sz - 1:h + self._sz, w:w + self._sz]

# gradient near the boundary of one patch

    def get_lgradient(self, row_idx, col_idx):
        h, w = self.patch_location(row_idx, col_idx)
        Nd = len(self._data.shape)
        return torch.diff(self._data[..., h:h + self._sz, w:w + 2], dim=Nd - 1)

    def get_rgradient(self, row_idx, col_idx):
        h, w = self.patch_location(row_idx, col_idx)
        Nd = len(self._data.shape)
        return torch.diff(self._data[..., h:h + self._sz, w + self._sz - 2:w + self._sz], dim=Nd - 1)

    def get_tgradient(self, row_idx, col_idx):
        h, w = self.patch_location(row_idx, col_idx)
        Nd = len(self._data.shape)
        return torch.diff(self._data[..., h:h + 2, w:w + self._sz], dim=Nd - 2)

    def get_bgradient(self, row_idx, col_idx):
        h, w = self.patch_location(row_idx, col_idx)
        Nd = len(self._data.shape)
        return torch.diff(self._data[..., h + self._sz - 2:h + self._sz, w:w + self._sz], dim=Nd - 2)


# gradient at the boundary of two patches.

    def get_lneighbor_gradient(self, row_idx, col_idx):
        h, w = self.patch_location(row_idx, col_idx)
        Nd = len(self._data.shape)
        return torch.diff(self._data[..., h:h + self._sz, w - 1:w + 1], dim=Nd - 1)

    def get_rneighbor_gradient(self, row_idx, col_idx):
        h, w = self.patch_location(row_idx, col_idx)
        Nd = len(self._data.shape)
        return torch.diff(self._data[..., h:h + self._sz, w + self._sz - 1:w + self._sz + 1], dim=Nd - 1)

    def get_tneighbor_gradient(self, row_idx, col_idx):
        h, w = self.patch_location(row_idx, col_idx)
        Nd = len(self._data.shape)
        return torch.diff(self._data[..., h - 1:h + 1, w:w + self._sz], dim=Nd - 2)

    def get_bneighbor_gradient(self, row_idx, col_idx):
        h, w = self.patch_location(row_idx, col_idx)
        Nd = len(self._data.shape)
        return torch.diff(self._data[..., h + self._sz - 1:h + self._sz + 1, w:w + self._sz], dim=Nd - 2)

    def get_ch0_offset(self, row_idx, col_idx):
        pass

    def get_data(self):
        return self._data.cpu().numpy().copy()

    def get_output(self):
        data = self.get_data()
        for row_idx in range(self._N):
            for col_idx in range(self._N):
                h, w = self.patch_location(row_idx, col_idx)
                data[..., 0, h:h + self._sz, w:w + self._sz] += self.get_ch0_offset(row_idx, col_idx)
                data[..., 1, h:h + self._sz, w:w + self._sz] -= self.get_ch0_offset(row_idx, col_idx)
        return data