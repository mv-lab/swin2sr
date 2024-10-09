from predtiler.dataset import get_tiling_dataset, get_tile_manager
from predtiler.tile_manager import tile_manager
from predtiler.tile_stitcher import stitch_predictions
from data_loader.biosr_dataset import BioSRDataLoader
from torch.utils.data import DataLoader
import torch

from tests.training import Swin2SRModule
from configs.biosr_config import get_config
import numpy as np


def tiled_prediction(patch_size=256, tile_size = 256, data_shape=(1004,1004)):
    patch_size = patch_size
    tile_size = tile_size
    data_shape = data_shape # size of the data you are working with
    manager = get_tile_manager(data_shape=data_shape, tile_shape=(1,tile_size,tile_size), 
                                patch_shape=(1,patch_size,patch_size))

        
    dset_class = get_tiling_dataset(BioSRDataLoader, manager)
    dataset = dset_class(root_dir='/group/jug/ashesh/data/BioSR/', 
                                    resize_to_shape=None,
                                    transform=None,
                                    noisy_data=True,
                                    noise_factor=1000, 
                                    gaus_factor=2000,
                                    )
    return dataset


if __name__ == '__main__':
    dset = tiled_prediction()
    test_loader = DataLoader(dset, batch_size=2, shuffle=False, num_workers=4)
    predictions = []
    
    
    
    for i in range(len(test_loader)):
        inp = test_loader.dataset[0][0]
        inp = torch.Tensor(inp)[None,None]
        config = get_config()
        model = Swin2SRModule(config)
        model.load_state_dict(torch.load('/home/michele.prencipe/tesi/transformer/swin2sr/logdir/zkvyvgz7swin2sr'))

        inp = inp.cuda()
        pred = model(inp)
        predictions.append(pred[0].numpy())

    predictions = np.stack(predictions) # shape: (number_of_patches, C, patch_size, patch_size)
    stitched_pred = stitch_predictions(predictions, dset.tile_manager)