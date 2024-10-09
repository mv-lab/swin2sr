import pytorch_lightning as pl
import os
import torch
import torch.nn as nn
import wandb
import yaml
import socket
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import display

from pytorch_lightning.callbacks import TQDMProgressBar

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from models.network_swin2sr import Swin2SR
from torch.utils.data import random_split
from data_loader.biosr_dataset import BioSRDataLoader
from configs.biosr_config import get_config
from models.swin2sr import Swin2SRModule
from utils.utils import Augmentations
from utils.utils import set_global_seed
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


set_global_seed(42)


def create_dataset(config, datadir, kwargs_dict=None,
                   transform = None, noisy_data = True, noisy_factor = 1000, gaus_factor = 1000, tiling_manager = None):
    if kwargs_dict is None:
        kwargs_dict = {}    
    resize_to_shape = (1004, 1004)
    
    
    if transform is not None:
        transform = Augmentations()        
    
    dataset = BioSRDataLoader(root_dir=datadir, 
                            resize_to_shape=resize_to_shape,
                            transform=transform,
                            noisy_data=noisy_data,
                            noise_factor=noisy_factor, 
                            gaus_factor=gaus_factor,
                            )
    
    train_ratio, val_ratio = 0.8, 0.1
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
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
        "epochs": config.training.num_epochs,
        "size": config.data.image_size,
        "poisson": config.data.poisson_noise_factor,
        "gauss": config.data.gaussian_noise_factor
    }
    config_str = f"LR: {args['learning_rate']}, Augmentations: True, Noisy_data: True, Noisy:{args['poisson']}, Gaus:{args['gauss']}" 
       
    
    node_name = os.environ.get('SLURMD_NODENAME', socket.gethostname())  
    
    # Initialize WandbLogger with a custom run name
    wandb_logger = WandbLogger(save_dir=logdir, project="SwinTransformer", name=f"{node_name}" + config_str)
    
    wandb_logger.experiment.config.update(config.to_dict())   
    model = Swin2SRModule(config)
    
    run_id = wandb_logger.experiment.id
    
    # Define two callback functions for early stopping and learning rate reduction
    model_filename = f'{run_id}swin2sr_epoch_{args["epochs"]}_image_size_{args["size"]}, '

    early_stopping = EarlyStopping(
        monitor='val_loss',  
        patience=50,    # How long to wait after last improvement
        #restore_best_weights=True,  # Automatically handled by PL's checkpoint system
        mode='min')
    
        # Define ModelCheckpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  
        save_top_k=1, 
        save_last=True,
        mode='min',         
        dirpath='/home/michele.prencipe/tesi/transformer/swin2sr/logdir',     
        filename=model_filename + '_best',
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = model_filename + "_last"
    callbacks=[early_stopping, checkpoint_callback]
    
    # Define the Trainer
    trainer = pl.Trainer(
        max_epochs=400,
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        precision=16,
        enable_progress_bar= True,
        callbacks = callbacks
    )    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Log metrics and model filename
    psnr1 = trainer.callback_metrics.get("val_psnr channel 1", None)
    psnr2 = trainer.callback_metrics.get("val_psnr channel 2", None)
    val_loss = trainer.callback_metrics.get("val_loss", None)
    
    
    wandb_logger.experiment.summary["Performance_Section"] = f"""
    ### Run ID: {run_id}
    **Configuration:** {config_str}

    #### Performance Metrics:
    - **PSNR channel 1**: {psnr1}
    - **PSNR channel 2**: {psnr2}
    - ** val_loss **: {val_loss}
    """
    
    model_filename = f'{run_id}swin2sr'
    wandb_logger.experiment.summary['model_weights_filename'] = model_filename
    
    # Save model
    saving_dir = "/home/michele.prencipe/tesi/transformer/swin2sr/logdir"
    model_filepath = os.path.join(saving_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)
    
    # Save wandb summary to a file
    summary_dict = dict(wandb_logger.experiment.summary)
    
    wandb.finish()
    
    data ={}
            
    # Define the JSON file path
    json_file_path = 'run_data.json'

        # Check if the JSON file exists
    if os.path.exists(json_file_path):
        # If the file exists, read its content
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

    # Append the new run data to the existing list
    data[f'{run_id}'] = summary_dict

    # Write the updated data back to the file
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == '__main__':
    logdir = 'tesi/transformer/swin2sr/logdir'
    wandb.login()
    config = get_config()
    
    dataset, train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = create_dataset(
        config=config, 
        datadir='/group/jug/ashesh/data/BioSR/',
        transform=True, noisy_data= True, 
        noisy_factor=1000, 
        gaus_factor=2000
    )
    create_model_and_train(config=config, 
                           logger=wandb, 
                           train_loader=train_loader, 
                           val_loader=val_loader,
                           logdir=logdir)
