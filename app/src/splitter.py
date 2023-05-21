import os
import cv2
import numpy as np
import mediapipe as mp
import pyrootutils
import hydra
from omegaconf import OmegaConf, DictConfig


# pyrootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)


from app.src.annotator import Annotator
from app.src.detector import FaceDetector
from app.src.filter import FacialFilter

class VideoParser:
    def __init__(self,
                 data_dir,
                 stride,
                 detector: FaceDetector, #= FaceDetector(1, mp_face_detector.FaceDetection(model_selection=0,min_detection_confidence=0.5)),
                 annotator: Annotator, #= Annotator(),
                 filter: FacialFilter #= FacialFilter("app/asset/filter/base.txt", ["app/asset/filter/image"])
                 ) -> None:
        self.parser: cv2.VideoCapture = cv2.VideoCapture()
        self.data_dir = data_dir
        self.video_name = ""
        self.stride = stride

        self.detector = detector
        self.annotator = annotator
        self.index = 0
        

        self.filter = filter
        self.filter.switch(0)


        #landmarks properties
        self.boxes = []
        self.landmarks = None
        self.target_lm = None
        self.velocity = 0

        print("System ready!!!")
        pass

    def feed(self, filename):
        self.video_name = filename[0: len(filename) - 4]
        file_path = os.path.join(self.data_dir, filename)
        self.parser.open(file_path)
        if not self.parser.isOpened():
            print("Cannot open video")
            exit(1)
        return {"name": self.video_name}
    

    def camera(self, index):
        self.parser = cv2.VideoCapture(index)
        self.parser.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.parser.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # self.parser.set(cv2.CAP_PROP_FPS, 12)
        while self.parser.isOpened() :
            retval, frame = self.parser.read()
            if retval == False:
                continue

            frame = self.execute_frame(frame)
            cv2.imshow('frame',frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        self.parser.release()
        cv2.destroyAllWindows()
        self.reset()

    # When deployed on website, what we retrieves is the frame from front-end, handle it seperately
    # Btw, we dont know the color-channel so errmmm, goodluck, will adjust later
    def execute_frame(self, frame):
        self.index += 1
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image = frame
        if self.index % self.stride == 0:
            detected, self.boxes, faces = self.detector.face_detect(image)
            
            if not detected:
                self.landmarks = None
                self.target_lm = None
                self.velocity = 0
            else:
                bounding_box = np.array([[max(0,box[0]), max(0,box[1])] for box in self.boxes])
                ratio = np.array([[box[2], box[3]] for box in self.boxes])
                self.target_lm = self.annotator.annotate(faces, np.zeros(bounding_box.shape), np.ones(ratio.shape))
                if (self.landmarks is None)  or (self.landmarks.shape[0] != self.target_lm.shape[0]):
                    self.landmarks = self.target_lm.copy()
                    self.velocity = 0
                else: 
                    self.velocity = (self.target_lm - self.landmarks) / self.stride
 

        if (self.landmarks is not None):
            self.landmarks += self.velocity
            image = self.filter.process(frame, self.landmarks, self.boxes, transform=True)
        else:
            return frame
        return image
    
    # to process on single frame
    def process_image(self, image):
        detected, boxes, faces = self.detector.face_detect(image)
        if not detected:
            return image
        
        bounding_box = np.array([[max(0,box[0]), max(0,box[1])] for box in self.boxes])
        ratio = np.array([[box[2], box[3]] for box in self.boxes])
        lm = self.annotator.annotate(faces, np.zeros(bounding_box.shape), np.ones(ratio.shape))

        return self.filter.process(image, lm, boxes, transform=True)
    def parse_video(self):
        output_dir = os.path.join(os.path.join(self.data_dir, "output"), self.video_name)
        os.makedirs(output_dir)
        index = 0
    
    def reset(self):
        self.index = 0

        #landmarks properties
        self.boxes = []
        self.landmarks = None
        self.target_lm = None
        self.velocity = 0
            
if __name__ == "__main__":

    config_path = os.path.join(os.environ["PROJECT_ROOT"], "configs")
    print(config_path)
    @hydra.main(version_base="1.3", config_path=config_path, config_name="app.yaml")
    def main(cfg: DictConfig):
        # print(OmegaConf.to_yaml(cfg))
        FacialFilter.setup_base(cfg.path.filter_base)
        capturer: VideoParser = hydra.utils.instantiate(cfg.splitter)
        capturer.camera(0)
    main()