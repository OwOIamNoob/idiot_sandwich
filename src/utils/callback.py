import torch
import wandb
import pytorch_lightning as pl

from src.data.dlib_datamodule import TransformDataset

class ImagePredictionLogger(pl.Callback):
    def __init__(self, 
                 val_samples, 
                 num_samples=16):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]
          
    def on_validation_epoch_end(self,
                                trainer,
                                pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        preds= pl_module(val_imgs)
        preds = torch.tensor.numpy(preds)

        trainer.logger.experiment.log({
            "examples": [wandb.Image() 
                            for x in TransformDataset.annotate_tensor(val_imgs, preds)], #zip(val_imgs, preds, self.val_labels)
            "global_step": trainer.global_step
            })