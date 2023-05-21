import os
import pyrootutils
import hydra
from omegaconf import OmegaConf, DictConfig

# pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from app.src.splitter import VideoParser
from app.src.filter import FacialFilter

config_path = os.path.join(os.environ["PROJECT_ROOT"], "configs")
processor: VideoParser 

@hydra.main(version_base="1.3", config_path= config_path, config_name="app.yaml")
def setup(cfg: DictConfig):
    FacialFilter.setup_base(cfg.path.filter_base)
    processor = hydra.utils.instantiate(cfg.splitter)
    print(type(processor))

def process_image(image):
    return processor.process_image(image)

# if the the image input is a part of a series :) useful for realtime
def process_frame(image):
    return processor.execute_frame(image)

def reset():
    processor.reset()

