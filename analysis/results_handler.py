import json
import os
import pickle

from core.data_split_type import DataSplitType
from core.tiff_reader import save_tiff


class PaperResultsHandler:

    def __init__(
        self,
        output_dir,
        eval_datasplit_type,
        patch_size,
        grid_size,
        mmse_count,
        skip_last_pixels,
        predict_kth_frame=None,
    ):
        self._dtype = eval_datasplit_type
        self._outdir = output_dir
        self._patchN = patch_size
        self._gridN = grid_size
        self._mmseN = mmse_count
        self._skiplN = skip_last_pixels
        self._predict_kth_frame = predict_kth_frame

    def dirpath(self):
        return os.path.join(
            self._outdir,
            f'{DataSplitType.name(self._dtype)}_P{self._patchN}_G{self._gridN}_M{self._mmseN}_Sk{self._skiplN}')

    @staticmethod
    def get_fname(ckpt_fpath):
        assert ckpt_fpath[-1] != '/'
        basename = '_'.join(ckpt_fpath.split("/")[4:]) + '.pkl'
        basename = 'stats_' + basename
        return basename

    @staticmethod
    def get_pred_fname(ckpt_fpath):
        assert ckpt_fpath[-1] != '/'
        basename = '_'.join(ckpt_fpath.split("/")[4:]) + '.tif'
        basename = 'pred_' + basename
        return basename

    def get_output_dir(self):
        outdir = self.dirpath()
        if self._predict_kth_frame is not None:
            outdir = os.path.join(outdir, f'kth_{self._predict_kth_frame}')

        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        return outdir

    def get_output_fpath(self, ckpt_fpath):
        outdir = self.get_output_dir()
        output_fpath = os.path.join(outdir, self.get_fname(ckpt_fpath))
        return output_fpath

    def save(self, ckpt_fpath, ckpt_stats):
        output_fpath = self.get_output_fpath(ckpt_fpath)
        with open(output_fpath, 'wb') as f:
            pickle.dump(ckpt_stats, f)
        print(f'[{self.__class__.__name__}] Saved to {output_fpath}')
        return output_fpath

    def dump_predictions(self, ckpt_fpath, predictions, hparam_dict):
        fname = self.get_pred_fname(ckpt_fpath)
        fpath = os.path.join(self.get_output_dir(), fname)
        save_tiff(fpath, predictions)
        print(f'Written {predictions.shape} to {fpath}')
        hparam_fpath = fpath.replace('.tif', '.json')
        with open(hparam_fpath, 'w') as f:
            json.dump(hparam_dict, f)

    def load(self, output_fpath):
        assert os.path.exists(output_fpath)
        with open(output_fpath, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    output_dir = '.'
    patch_size = 23
    grid_size = 16
    mmse_count = 1
    skip_last_pixels = 0

    saver = PaperResultsHandler(output_dir, 1, patch_size, grid_size, mmse_count, skip_last_pixels)
    fpath = saver.save('/home/ashesh.ashesh/training/disentangle/2210/D7-M3-S0-L0/82', {'a': [1, 2], 'b': [3]})

    print(saver.load(fpath))
