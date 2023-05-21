import gradio as gr
import os
import hydra
import pyrootutils
from omegaconf import OmegaConf, DictConfig

pyrootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)
from app.src.splitter import VideoParser
from app.src.filter import FacialFilter



config_path = os.path.join(os.environ["PROJECT_ROOT"], "configs")

global processor

def prepare(cfg: DictConfig):
    FacialFilter.setup_base(cfg.path.filter_base)


@hydra.main(version_base="1.3", config_path= config_path, config_name="app.yaml")
def setup(cfg: DictConfig):
    # FacialFilter.setup_base(cfg.path.filter_base)
    prepare(cfg)
    global processor
    processor = hydra.utils.instantiate(cfg.splitter)
    print(type(processor))

def process_image(image):
    return processor.process_image(image)

# if the the image input is a part of a series :) useful for realtime
def process_frame(image):
    return processor.execute_frame(image)

def reset():
    processor.reset()


# Create VideoParser
setup()

interaction_tab = gr.Interface(fn=process_frame,
                               inputs=gr.Image(source="webcam", streaming=True, shape=[640, 480], type="numpy"),
                               outputs=gr.Image(type="numpy"),
                               title="Filter app",
                               description = "OwO I dont know what im doing",
                               article= "Created by a noob",
                               live=True)

interaction_tab.launch(server_name="localhost", server_port=8000)