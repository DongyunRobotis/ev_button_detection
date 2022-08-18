from contextlib import closing

from cv2 import dft
from matplotlib import transforms
from scipy.fftpack import dst
from .vision.transforms import Normalize, Compose, Normalize_Gray, Resize
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import os
import math
from cmath import nan

class Matching:
    def __init__(self, path):
        self.pattern_image = cv.imread(path)
        self.pattern_list ={
            'Panel': [254, 460], 
            'Call': [253, 94], 
            4: [157, 271], 
            10: [347, 271], 
            3: [158, 354], 
            9: [349, 355], 
            2: [152, 432], 
            1: [155, 515], 
            8: [346, 436], 
            7: [345, 517], 
            6: [344, 598], 
            81: [153, 598], 
            82: [154, 681], 
            83: [153, 762], 
            5: [347, 682], 
            'Open': [153, 926], 
            'Close': [344, 925]}
        self.pattern_pair = [
          ['Open', 'Close'],
          [82, 5],
          [81, 6],
          [1, 7],
          [2, 8],
          [3, 9],
          [4, 10]
        ]
        self.compare_image = cv.imread('pattern_image.jpg')
    
    def calculate_matching_pts(self, class_pts):
        com_pattern_list = []
        com_result_list = []
        class_pts.sort(key=lambda x:x[0])
        
        for rcp in class_pts:
            if rcp[0] in self.pattern_list:
                com_pattern_list.append([rcp[0], self.pattern_list[rcp[0]][0], self.pattern_list[rcp[0]][1]])
                com_result_list.append([rcp[0], rcp[1], rcp[2]])
        return com_pattern_list, com_result_list
    
    def make_pattern_image(self):
        pannel_pts = [[361,86],[792,76],[363,1378],[729,1408]]
        pannel2_pts = [[0, 0], [500, 0], [0 , 1000], [500, 1000]]
        s = np.array(pannel_pts).reshape(-1, 1, 2).astype(np.float32)
        d = np.array(pannel2_pts).reshape(-1, 1, 2).astype(np.float32)
        P = cv.getPerspectiveTransform(s, d)
        self.pers_pattern_image = cv.warpPerspective(self.pattern_image, P, (500,1000))
        return self.pers_pattern_image
    
    def calculate_homography_matrix(self, compare_pattern_list, compare_result_list):
        cpl = []
        crl = []
        for n in range(len(compare_pattern_list)):
            cpl.append([compare_pattern_list[n][1],compare_pattern_list[n][2]])
            crl.append([compare_result_list[n][1],compare_result_list[n][2]])  
        
        src = np.array(cpl).reshape(-1, 1, 2).astype(np.float32)
        tar = np.array(crl).reshape(-1, 1, 2).astype(np.float32)
        self.homo_matrix, self.matching_INDEX = cv.findHomography(src, tar, cv.RANSAC, maxIters=100)
        
        if self.homo_matrix is not nan:
            if self.homo_matrix is not None:
                H_D = self.homo_matrix[0][0]*self.homo_matrix[1][1] - self.homo_matrix[0][1] * self.homo_matrix[1][0]
                H_sx = math.sqrt(self.homo_matrix[0][0]**2 + self.homo_matrix[1][0]**2)
                H_sy = math.sqrt(self.homo_matrix[0][1]**2 + self.homo_matrix[1][1]**2)
                H_P = math.sqrt(self.homo_matrix[2][0]**2 + self.homo_matrix[2][1]**2)
                mat_key = []
                for i in range(len(self.matching_INDEX)):
                    if self.matching_INDEX[i] == 1:
                        mat_key.append(compare_pattern_list[i][0])
                self.matching_key = mat_key
                if H_D <= 0 :
                    return False
                elif H_sx < 0.2:
                    return False
                elif H_sx > 3:
                    return False
                elif H_sy < 0.2:
                    return False
                elif H_sy > 3:
                    return False
                elif H_P > 0.001:
                    return False
                else :
                    return True
            else :
                return False
        else :
            return False
        
    def apply(self):
        new_pattern = {}
        #img = np.asarray(self.compare_image, dtype=np.float32)
        #class_homo_img = cv.warpPerspective(img, self.homo_matrix, (ori_image.shape[1], ori_image.shape[0]))
        #Mix Original Img & Class_Homo_Img
        #Mix_img = cv.addWeighted(ori_image, 0.5, class_homo_img, 0.5, 0, dtype = cv.CV_8UC3)
        
        for key in self.pattern_list:
            [x, y] = self.pattern_list[key]
            p = np.array((x, y, 1)).reshape((3,1))
            temp_p = self.homo_matrix.dot(p)
            sum = np.sum(temp_p,1)
            px = int(round(sum[0]/sum[2]))
            py = int(round(sum[1]/sum[2]))
            #cv.circle(Mix_img, (px, py), 3, (0, 0, 255), -1)
            #text=str(key)
            #font=cv.FONT_HERSHEY_SIMPLEX
            #org=(px, py)
            #if key in self.matching_key:
            #    cv.putText(Mix_img,text,org,font,0.5,(0,255,0),2)
            #else :
            #    cv.putText(Mix_img,text,org,font,0.5,(255,0,0),2)
            new_pattern[key] = [px, py]
        
        #return Mix_img, new_pattern
        return new_pattern

    def calculate_reprojection_error(self, com_result_list, pattern_dict):
        rep_err = 0
        count = 0
        for n in range(len(self.matching_INDEX)):
            if self.matching_INDEX[n][0] == 1:
                [c, x, y] = com_result_list[n]
                d_x = x - pattern_dict[c][0]
                d_y = y - pattern_dict[c][1]
                rep_err += math.sqrt(d_x**2 + d_y**2)
                count += 1
        rep_err /= count
        return rep_err

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def preprocess(image_path, w, h, rgb):
    transforms = Compose([
        Resize((w, h)),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ],
        to_rgb=rgb)
    return np.expand_dims(transforms(image_path), axis = 0)

def preprocess_cls(image_path, w, h, rgb):
    transforms = Compose([
        Resize((w, h)),
        Normalize_Gray(mean=0.5, std=0.5)
        ],
        to_rgb=rgb,
        )
    return np.expand_dims(transforms(image_path), axis = 0)

def get_color_map_list(num_classes):
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]
    return color_map

def visualize(image, result, save_dir=None, weight=0.2):
    color_map = get_color_map_list(256)
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = np.array(color_map).astype("uint8")
    c1 = cv.LUT(result, color_map[:, 0])
    c2 = cv.LUT(result, color_map[:, 1])
    c3 = cv.LUT(result, color_map[:, 2])
    pseudo_img = np.dstack((c1, c2, c3))
    im = image.copy()
    im = cv.resize(im, (512, 512))
    vis_result = cv.addWeighted(im, weight, pseudo_img, 1 - weight, 0)
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_name = os.path.split(image)[-1]
        out_path = os.path.join(save_dir, image_name)
        cv.imwrite(out_path, vis_result)
    else:
        return vis_result


def img_check_show(img):
    cv.imshow('Test', img)
    cv.waitKey(0)

def orb_matching(l_img, r_img):
    orb = cv.ORB_create()
    kps1, des1 = orb.detectAndCompute(l_img, None)
    kps2, des2 = orb.detectAndCompute(r_img, None)
    bf = cv.BFMatcher_create()
    matches = bf.knnMatch(des1, des2, k = 2)
    good = []
    for m, n in matches : 
        if m.distance < 0.4 * n.distance:
            good.append([m])
    np.random.shuffle(good)
    image_match = cv.drawMatchesKnn(l_img, kps1, r_img, kps2, good[:10], flags=2, outImg = l_img)
    return l_img