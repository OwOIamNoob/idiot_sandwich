from cv2 import INTER_CUBIC
import mediapipe as mp
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pyrootutils


mp_face_detector = mp.solutions.face_detection


pyrootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)

class FaceDetector:
    def __init__(self, 
                 resize=1, 
                 model = mp_face_detector.FaceDetection(model_selection=0,min_detection_confidence=0.5),
                 ):
        self.resize = resize
        self.detector = model
        print("Detector initialized")
        
    def face_detect(self,image):
        """Detect faces in frames using strided ."""
        
        
        if self.resize != 1:
            image = cv2.resize(image,(int(image.shape[1] * self.resize), int(image.shape[0] * self.resize)), interpolation= cv2.INTER_CUBIC)
        detected = False
        image = cv2.copyMakeBorder(image, 
                                   top=20, 
                                   bottom=20,
                                   left=20,
                                   right=20,
                                   borderType=cv2.BORDER_CONSTANT,
                                   value=[255, 255, 255])
        detections = self.detector.process(image).detections
        boxes = []
        if detections is None:
            detected = False
            return detected, [], []
        for detection in detections:
            boxes.append([detection.location_data.relative_bounding_box.xmin * image.shape[1],
                          detection.location_data.relative_bounding_box.ymin * image.shape[0],
                          detection.location_data.relative_bounding_box.width * image.shape[1],
                          detection.location_data.relative_bounding_box.height * image.shape[0]])
        
        boxes = np.array(boxes)
        
        detected = True
        faces = []
        
    
        for box in boxes:
            # print(box)
            box = [ max(0, int(b)) for b in box]
            faces.append(image[box[1]:(box[3] + box[1]), 
                               box[0]:(box[2] + box[0])])
            # print(len(faces))

        # changed back to original size
        # the image can be shrinked, but the bounding box must be original. - 
        # 'cause it's explicit and doesnt affect the augmentation
        boxes[:, 1] -= 20
        boxes[:, 0] -= 20
        boxes = boxes / self.resize
        return detected, boxes, faces
    
    #draw on mat
    @staticmethod
    def show_bounding_box(image, boxes):
        for box in boxes:
            image = cv2.rectangle(image, [int(box[0]), int(box[1])], [int(box[2]), int(box[3])], color=(0, 255, 0), thickness=15)
        
        return image

if __name__ == "__main__":
    import os
    import hydra
    from omegaconf import OmegaConf, DictConfig
    
    config_path = os.path.join(os.environ["PROJECT_ROOT"], "configs")
    @hydra.main(version_base="1.3", config_path=config_path, config_name="app.yaml")
    def main(cfg: DictConfig):
        
        detector = hydra.utils.instantiate(cfg.splitter.detector)
        print(type(detector))
        # # print(OmegaConf.to_yaml(cfg))
        # detector: FaceDetector = hydra.utils.instantiate(cfg.detector)
        # annotator: Annotator = hydra.utils.instantiate(cfg.annotator)
        # filter: FacialFilter = FacialFilter("app/asset/filter/base.txt", ["app/asset/filter/image/mosiac.png"])
        # # filter_img =cv2.cvtColor(cv2.imread("app/asset/filter\image\mosiac.png", cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        # # filter.feed(filter_img)
        # filter.switch(0)
        
        # if detector is None:
        #     print("Failed to fetch")
        #     exit(1)
        
        # image = cv2.cvtColor(cv2.imread("app/asset/bound_test.jpg"), cv2.COLOR_BGR2RGB)
        # print(image.shape)
        # detected, boxes, faces = detector.face_detect(image[:, :, 0:3])
        # print(boxes)
        # print(type(faces))
        # bounding_box = np.array([[box[0], box[1]] for box in boxes])
        # ratio = np.array([[box[2], box[3]] for box in boxes])
        # pred = annotator.annotate(faces, bounding_box, ratio, transform=True)
        # triangles, box = FacialFilter.get_triangles(pred, ratio, transform=False)
        # print("Box: {}".format(box))
        # image = FacialFilter.show_triangles(image, triangles, box[0])
        # cv2.imwrite("app/asset/output/output.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # plt.imshow(image)
        # plt.show()
    main()