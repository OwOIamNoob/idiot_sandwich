import os
import sys
from typing import Dict
import cv2
import hydra
import numpy as np
import pyrootutils
import omegaconf
from omegaconf import DictConfig

# pyrootutils.setup_root(search_from=__file__, indicator=".project-root",pythonpath=True)
sys.path.append("app/src/filter.py")
# This code is not optimized, the optimized way is to predict Delaunay for every frame. 
# :) But my computer sucks so I used predicated triangles

class FacialFilter:

    #using finetuned triangles 
    list_of_triangles = []
    
    def __init__(self, path):
        self.lm = []
        self.triangles = []
        self.image = []
        self.load(path)
        self.index = 0
        print("Filter created")
    
    def load(self, path):
        if not os.path.isfile(path):
            for folder in sorted(os.listdir(path)):
                filter = os.path.join(path, folder)
                for file in sorted(os.listdir(filter)):
                    img_path = os.path.join(filter, file)
                    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if image is None:
                        self.lm.append(np.loadtxt(img_path))
                    else:
                        self.image.append(cv2.imread(img_path, cv2.IMREAD_UNCHANGED))
        else:
            image = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
            self.image.append(image)
        
        print(len(self.image))
        print(self.lm[0].shape)
    
    @staticmethod
    def setup_base(path):
        FacialFilter.list_of_triangles = np.loadtxt(path, dtype=np.int8)
    
    def switch(self, index):
        self.index = (self.index + index) % len(self.image) 
        self.setup(self.index)

    def feed(self, img):
        self.image.append(img)

    def setup(self, index):
        h, w, _ = self.image[index].shape
        self.triangles = np.array([[self.lm[index][triangle[0]], self.lm[index][triangle[1]], self.lm[index][triangle[2]]] 
                            for triangle in FacialFilter.list_of_triangles]) * [h, w]
        # to get the actual position of the triangle
        self.triangles = self.triangles.astype(int)
        
    
    # return scaled set of triangles
    # the triangles is processed in [0, size] dimension, no translation because we're on image processing task
    @staticmethod
    def get_triangles(lms, size, *, transform=False):
        output = []
        lm_boxes = []
        for i in range(lms.shape[0]):
            lm = lms[i].copy()
            if transform:
                lm = lm * size[i].astype(int)

            lm = lm.astype(int)
            lm_box = np.array(cv2.boundingRect(lm), dtype=float)
            lm_box += [-0.1 * lm_box[2], -0.2 * lm_box[3], 0.2  * lm_box[2], 0.2 * lm_box[3]] 
            lm_box = np.ceil(lm_box).astype(int)
            lm = np.concatenate((lm, np.array([[lm_box[0], lm_box[1]],
                                                [lm_box[0] + lm_box[2] - 1, lm_box[1]],
                                                [lm_box[0] + lm_box[2] - 1, lm_box[1] + lm_box[3] - 1],
                                                [lm_box[0], lm_box[1] + lm_box[3] - 1]] )), axis=0)
            min_bound = np.array([lm_box[0], lm_box[1], 0, 0]).astype(int)
            lm -= min_bound[0:2]
            triangles = np.array([[lm[triangle[0]], lm[triangle[1]], lm[triangle[2]]] 
                                for triangle in FacialFilter.list_of_triangles])
            
            lm_boxes.append(lm_box)
            output.append(triangles)
        
        return np.array(output).astype(int), np.array(lm_boxes, dtype=int)
    
    # warp image from base triangles to target triangles
    # target coordinates is scaled but not translated, not normalized
    def warp_image(self, target, box):
        h, w, ch = self.image[self.index].shape
        output = np.zeros((box[3], box[2], 4), dtype=self.image[self.index].dtype)
        for i in range(self.triangles.shape[0]):
            r1 = cv2.boundingRect(self.triangles[i])
            r2 = cv2.boundingRect(target[i])
            tri1Cropped = []
            tri2Cropped = []
            for j in range(0, 3):
                tri1Cropped.append(((self.triangles[i][j][0] - r1[0]), (self.triangles[i][j][1] - r1[1])))
                tri2Cropped.append(((target[i][j][0] - r2[0]), (target[i][j][1] - r2[1])))
            img1Cropped = self.image[self.index][r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
            warpMat =  cv2.getAffineTransform(np.float32(tri1Cropped), np.float32(tri2Cropped))
            img2Cropped = cv2.warpAffine(img1Cropped, 
                                         warpMat, 
                                         (r2[2], r2[3]), 
                                         None, 
                                         flags=cv2.INTER_LINEAR, 
                                         borderMode=cv2.BORDER_TRANSPARENT)

            # use fixed point first
            mask = np.zeros((r2[3], r2[2], 4), dtype = np.float32)
            cv2.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0, 1.0), 16, 0);
            img2Cropped = img2Cropped * mask
     
            # Copy triangular region of the rectangular patch to the output image
            output[r2[1]:r2[1]+mask.shape[0], r2[0]:r2[0]+mask.shape[1]] = output[r2[1]:r2[1]+mask.shape[0], r2[0]:r2[0]+mask.shape[1]] * ( (1.0, 1.0, 1.0, 1.0) - mask )
     
            output[r2[1]:r2[1]+mask.shape[0], r2[0]:r2[0]+mask.shape[1]] = output[r2[1]:r2[1]+mask.shape[0], r2[0]:r2[0]+mask.shape[1]] + img2Cropped
        
        return output
    
    def process(self, image, preds, face_boxes, *, transform=False):
        set_triangles, ft_boxes = FacialFilter.get_triangles(preds, face_boxes[:, 2:4], transform=transform)
        for triangles, ft_box, face_box in zip(set_triangles[::-1], ft_boxes[::-1], face_boxes[::-1]):
            transformed = self.warp_image(triangles, ft_box)
            transformed = cv2.GaussianBlur(transformed, (3, 3), 10)
            image = FacialFilter.apply_filter(image, transformed, face_box, ft_box)
        
        return image
            
    @staticmethod
    def find_delaunay(lm):
        vertex_dict: Dict = dict()
        box = cv2.boundingRect(lm)
        subdiv = cv2.Subdiv2D()
        subdiv.initDelaunay(box)
        
        for index in range(lm.shape[0]):
            vertex_dict[subdiv.insert(lm[index])] = index
        print(vertex_dict)

        edgesID = subdiv.getLeadingEdgeList()
        edges = []
        
        for edgeID in edgesID:
            orgID,_ = subdiv.edgeOrg(edgeID)
            dstID,_ = subdiv.edgeDst(edgeID)
            edges.append([orgID, dstID])
      
        return subdiv.getEdgeList()
    

    @staticmethod
    def show_triangles(image, triangle_list, box):
        if triangle_list is None:
            return image

        for triangles in triangle_list:
            for triangle in triangles:
                for i in range(0,3):
                    cv2.line(image, triangle[i] + box[0:2], triangle[(i+1) % 3] + box[0:2], thickness=1, color=(0, 255, 0))
        
        return image
        
    @staticmethod
    def show_delaunay(image, lms, size, translate, *, normalized=True):
        
        if lms is None:
            return image
    

        for i in range(lms.shape[0]):
            lm = lms[i].copy()
            lm = np.concatenate((lm, np.array([[translate[i][0], translate[i][1]],
                                               [translate[i][0] + size[i][0], translate[i][1]],
                                               [translate[i][0] + size[i][0], translate[i][1] + size[i][1]],
                                               [translate[i][0], translate[i][1] + size[i][1]]] )), axis=0)
            if normalized:
                lm = lm * size + translate
            lm = lm.astype(int)
            # print(lm)
            for triplet  in FacialFilter.list_of_triangles:
                cv2.line(image, lm[triplet[0]], lm[triplet[1]], thickness=1, color=(255, 0, 0))
                cv2.line(image, lm[triplet[0]], lm[triplet[2]], thickness=1, color=(255, 0, 0))
                cv2.line(image, lm[triplet[1]], lm[triplet[2]], thickness=1, color=(255, 0, 0))
        return image

    # find the box of overlaying image
    @staticmethod
    def find_overlay(box, ft_box, img_shape):
        # x, y, w, h
        mat = np.array([box[0] + ft_box[0], box[1] + ft_box[1], ft_box[2], ft_box[3]]).astype(int)
        bound = np.array([min(0, mat[0]), 
                          min(0, mat[1]), 
                          max(0, mat[0] + mat[2] - img_shape[1]), 
                          max(0, mat[1] + mat[3] - img_shape[0])]).astype(int)
        bound[2:4] -= bound[0:2]
        # print(bound)
        dst = mat - bound
        ft_box = np.array([0 - bound[0], 0 - bound[1], ft_box[2] - bound[2], ft_box[3] - bound[3]] , dtype=int)
        return dst, ft_box

    @staticmethod
    def apply_filter(src, filter, src_box, ft_box):
        dst, ft = FacialFilter.find_overlay(src_box, ft_box, src.shape)
        # print(filter.shape)
        # print(dst, ft)
        alpha = filter[ft[1] : ft[1] + ft[3], ft[0] : ft[0] + ft[2], 3].astype(float) / 255
        alpha = cv2.merge((alpha, alpha, alpha))
        src[dst[1] : dst[1] + dst[3], dst[0] : dst[0] + dst[2]] = src[dst[1] : dst[1] + dst[3], dst[0] : dst[0] + dst[2]] * ((1.0, 1.0, 1.0) - alpha) + filter[ft[1] : ft[1] + ft[3], ft[0] : ft[0] + ft[2], 0:3] * alpha

        return src

    @staticmethod
    def apply_resized(src, filter, src_box):
        filter = cv2.resize(filter, src_box[2:4].astype(int))
        # x, y, w, h
        dst, ft = FacialFilter.find_overlay(src_box, [0, 0, src_box[2], src_box[3]], src.shape)
        alpha = filter[ft[1] : ft[1] + ft[3], ft[0] : ft[0] + ft[2], 3].astype(float) / 255
        alpha = cv2.merge((alpha, alpha, alpha))
        src[dst[1] : dst[1] + dst[3], dst[0] : dst[0] + dst[2]] = src[dst[1] : dst[1] + dst[3], dst[0] : dst[0] + dst[2]] * ((1.0, 1.0, 1.0) - alpha) + filter[ft[1] : ft[1] + ft[3], ft[0] : ft[0] + ft[2], 0:3] * alpha
        
        return src

if __name__ == "__main__":
    config_path = os.path.join(os.environ["PROJECT_ROOT"], "configs")
    print(config_path)

    @hydra.main(version_base="1.3", config_path=config_path, config_name="app.yaml")
    def main(cfg: DictConfig):
        filter = hydra.utils.instantiate(cfg.splitter.filter)
        # filter = FacialFilter("app/asset/filter/base.txt",
        #                       "app/asset/filter/image")
        print(type(filter))
    
    main()
