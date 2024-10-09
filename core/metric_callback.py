"""
Custom class to track a metric and call a specific function when the criterion is fullfilled.
"""
from pytorch_lightning.callbacks import Callback


class ValMetricCallback(Callback):
    def __int__(self, mode, callback_fn):
        super().__init__()
        assert mode in ['min', 'max']

    # def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
    #     import pdb
    #     pdb.set_trace()

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        psnr = trainer.callback_metrics['val_psnr']
