import os
from typing import List, Dict
import PIL
from PIL import ImageDraw
import cv2
from torch import Tensor
import torch
import torchvision
import numpy as np
import pyrootutils
import hydra
from omegaconf import OmegaConf, DictConfig

# pyrootutils.setup_root(search_from=__file__, indicator="setup.cfg", pythonpath=True)

class Annotator:

    def __init__(self,
                 net: torch.nn.Module,
                 ckpt_path,
                 transform: torchvision.transforms.transforms,
                 dimension: List) -> None:
        self.module = net
        print(ckpt_path)
        self.module.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.transform = transform 
        self.dimension = torch.Tensor(dimension)
        print("Annotator initialized")
        pass

    @torch.no_grad()
    def annotate(self, x, bounding_box, ratio, *, transform=False):
        input = []
        # assuming that input is 'faces' = list 
        for i in range(len(x)):
            # in case x is tensor :))
            print("Image size:" + str(x[i].shape))
            # only use RGB
            transformed = self.transform(image=np.array(x[i][:, :, 0:3]))
            input.append(transformed["image"])
        
        # convert to tensor
        input = torch.stack(input)
        print(input.size())
        # pred is normalize between [-0.5, 0.5] in the [224, 224] window
        pred = self.module.forward(input).detach().numpy()
        for i in range(pred.shape[0]):
            if transform:
                pred[i] = (pred[i] + 0.5) * ratio[i] + bounding_box[i] 
            else:
                pred[i] += 0.5
        return pred

    @staticmethod 
    def annotate_image(self,
                       image: PIL.Image, 
                       landmarks: Tensor):
        draw = ImageDraw.Draw(image)
        landmarks = Tensor.numpy(landmarks)
        for i in range(landmarks.shape[0]):
            draw.ellipse((landmarks[i, 0] - 2, landmarks[i, 1] - 2,
                          landmarks[i, 0] + 2, landmarks[i, 1] + 2), fill=(255, 0, 0))
        return image
    
    @staticmethod
    def show_landmarks(image,
                       landmarks: list):
        for i in range(len(landmarks)):
            for index in range(len(landmarks[i])):
                cv2.circle(image, (int(landmarks[i][index][0]), int(landmarks[i][index][1])), radius=3, color=(255, 0, 0), thickness=-1)
                cv2.putText(image,str(index),(int(landmarks[i][index][0]), int(landmarks[i][index][1])), fontFace=cv2.FONT_HERSHEY_PLAIN, color=(0, 0, 0), fontScale=0.5, thickness=1 )
        
        return image

if __name__ == "__main__":

    config_path = os.path.join(os.environ["PROJECT_ROOT"], "configs")
    print(config_path)
    @hydra.main(version_base="1.3", config_path=config_path, config_name="app.yaml")
    def main(cfg: DictConfig):
        # annotator: Annotator = hydra.utils.instantiate(cfg.app.annotator)
        
        # datamodule: DlibDataModule = hydra.utils.instantiate(cfg.data)
        # datamodule.setup()
        # test_dataloader = datamodule.val_dataloader()
        # data = next(iter(test_dataloader))
        # x, y = data
        # pred = annotator.forward_tensor(x, torch.Tensor([0, 0]), torch.Tensor([1, 1]))
        # images = TransformDataset.annotate(x, pred)
        # torchvision.utils.save_image(images, "C:/Lab/project/facial_landmarks-wandb/output/testing_datamodule_result.png")
        # print(images.shape)


        # path = cfg.app.annotator.ckpt_path
        # checkpoint = torch.load(path, map_location="cpu")
        # # print(checkpoint["state_dict"])
        # # model = DlibDataModule.load_from_checkpoint(path, net = simple_resnet.SimpleResnet)
        # model = simple_resnet.SimpleResnet()
        # state_dict = checkpoint['state_dict']
        # keys = [key.replace("net.","") for key in list(state_dict.keys())]
        # state_dict = dict(zip(list(state_dict.keys()), list(state_dict.values())))

        # model.load_state_dict(state_dict)

        # torch.save(state_dict, "F:/project/facial_landmarks-wandb/app/model/model.pth")
        path = "F:/project/facial_landmarks-wandb/app/model/model.pth"
        state_dict = torch.load(path, map_location="cpu")
        model = hydra.utils.instantiate(cfg.splitter.annotator.net)
        model.load_state_dict(state_dict)
        annotator = hydra.utils.instantiate(cfg.splitter.annotator)
        
        # print(checkpoint)
    main()
        
