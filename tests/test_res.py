import os
import hydra
from matplotlib import pyplot as plt
from omegaconf import DictConfig
import pyrootutils
from pytorch_lightning import LightningDataModule, LightningModule
import torch
import torchvision
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm



pyrootutils.setup_root(__file__,indicator=".project-root", pythonpath=True)
config_path = os.path.join(os.environ["PROJECT_ROOT"], "configs")

from src.data.dlib_datamodule import TransformDataset
from src.models.dlib_module import DlibLitModule


@hydra.main(version_base="1.3",config_path=config_path, config_name="train.yaml")
def main(cfg: DictConfig):
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    # model: LightningModule = hydra.utils.instantiate(cfg.model)
    datamodule.prepare_data()
    datamodule.setup()
    model = DlibLitModule.load_from_checkpoint(cfg.ckpt_path, net=hydra.utils.instantiate(cfg.model.net))
    test_data = datamodule.test_dataloader()
    data = next(iter(test_data))
    print(len(data))
    x, y = data
    _, preds, _ = model.model_step(data)
    preds = torch.Tensor.cpu(preds)
    preds = preds.detach().numpy()
    x = torch.Tensor.cpu(x)

    images = TransformDataset.annotate_tensor(x, preds)
    torchvision.utils.save_image(images, "C:/Lab/project/facial_landmarks-wandb/output/test_datamodule_result.png")
    print(images.shape)
    pass

if __name__ == "__main__":
    main()