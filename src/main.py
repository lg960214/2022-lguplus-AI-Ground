import os
import numpy as np
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from dataset import Uplus_DataModule
from models import MultiVAE
from builder import Engine


def cli_main(conf):
    pl.seed_everything(conf.DataModule.seed)

    uplus_dm = Uplus_DataModule(conf.DataModule)

    # if not os.path.exists(conf.DataModule.processed_data_dir):
    uplus_dm.setup()

    train_loader = uplus_dm.train_dataloader()
    valid_loader = uplus_dm.val_dataloader()

    model = MultiVAE(conf.Model)
    engine = Engine(model, conf.Trainer, conf.DataModule.val_te_mat_path)

    checkpoint_callback = ModelCheckpoint(
        filename="epoch{epoch}-val_score{valid_score:.4f}",
        monitor="valid_score",
        mode="max",
        auto_insert_metric_name=False,
    )

    earlystop_callback = EarlyStopping(
        monitor="valid_recall@25", patience=20, verbose=True, mode="max"
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, earlystop_callback],
        max_epochs=conf.Trainer.epochs,
        # For GPU Setup
        deterministic=torch.cuda.is_available(),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1,
        precision=16 if conf.Trainer.fp_16 else 32,
    )

    trainer.fit(engine, train_loader, valid_loader)


if __name__ == "__main__":
    conf = OmegaConf.load("./configs/vae.yaml")
    cli_main(conf)
