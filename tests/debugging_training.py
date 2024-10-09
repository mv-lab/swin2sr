import pytorch_lightning as pl
import os
import torch
import torch.nn as nn
import wandb
import yaml
import socket

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from models.network_swin2sr import Swin2SR
from torch.utils.data import random_split
from data_loader.biosr_dataset import BioSRDataLoader
from configs.biosr_config import get_config
from models.swin2sr import Swin2SRModule
from utils.utils import Augmentations


def create_dataset(config, datadir, kwargs_dict=None, noisy_data = False, noisy_factor = 0.1, resize_to_shape=(256,256)):
    if kwargs_dict is None:
        kwargs_dict = {}
        
    resize_to_shape = (256, 256)
    
    augmentations = Augmentations() 
    dataset = BioSRDataLoader(root_dir=datadir, resize_to_shape=resize_to_shape, transform=augmentations, noisy_data=noisy_data, noise_factor=noisy_factor)
    
    train_ratio, val_ratio = 0.8, 0.1
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    torch.manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    torch.manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4)
    
    return dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def create_model_and_train(config, logger, train_loader, val_loader, logdir): 
    args = {
        "learning_rate": config.training.lr,
        "architecture": config.model.model_type,
        "dataset": "BioSRDataset",
        "epochs": config.training.num_epochs
    }
    
    print(f"Learning rate: {args['learning_rate']}")
    
    model = Swin2SRModule(config)
    
    # Define the Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        log_every_n_steps=1500,
        check_val_every_n_epoch=1,
        precision=16,
        enable_progress_bar=True
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    psnr1 = trainer.callback_metrics.get("val_psnr channel 1", None)
    psnr2 = trainer.callback_metrics.get("val_psnr channel 2", None)
    val_loss = trainer.callback_metrics.get("val_loss", None)
    
    print("psnr1:", psnr1)
    print("psnr2:", psnr2)
    print("val_loss:", val_loss)


if __name__ == '__main__':
    logdir = 'tesi/transformer/swin2sr/logdir'
    config = get_config()    
    dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = create_dataset(
        config=config, datadir='/group/jug/ashesh/data/BioSR/', noisy_data= False, noisy_factor=0.1
    )
    create_model_and_train(config=config, logger=None, train_loader=train_loader, val_loader=val_loader, logdir=logdir)
