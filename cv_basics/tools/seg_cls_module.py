from cmath import nan
import enum
from typing_extensions import runtime
from unittest import result
from pytools import average
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2 as cv
import os
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from PIL import Image
from image_proc import preprocess, crop_patch, normalize, get_color_map_list, visualize, img_check_show, divide_button_img, Matching
from trt_proc import Segmentation, Classification
import pyzed.sl as sl
import math
import random
import pandas as pd

def convert_zed2mat(mat):
    a, b, c, d = cv.split(mat)
    cv_mat = cv.merge((a, b, c))
    return cv_mat

def eve_button_2D_detection(seg_module, cls_module, matching_module, img):
    #TensorRT Segmentation Inference
    inf_result = seg_module.infer(img)
    #Visualize Segmentation Result
    vis_seg_result = seg_module.result_image()
    #cv.imshow('vis_seg_result', vis_seg_result)
    #Binarize Button Class
    btn_image = seg_module.get_button_binary_image()
    seg_patch_image, seg_patch_bbox, seg_patch_center_pts = seg_module.crop_patch()
    #cv.imshow('btn_image', seg_module.input_image)
    #Classification
    class_bbox, class_patch_center_pts = cls_module.infer(seg_patch_image, seg_patch_bbox, seg_patch_center_pts)
    img = cls_module.make_class_image(img, class_bbox)
    #cv.imshow('class_image', class_img)
    #Match Seg_Points & Pattern Points
    compare_pattern_list, compare_result_list = matching_module.calculate_matching_pts(class_patch_center_pts)
    #Class Image Load
    class_pers_img = matching_module.make_pattern_image()
    if len(compare_pattern_list) > 3:
        if matching_module.calculate_homography_matrix(compare_pattern_list, compare_result_list):
            homo_result_image, new_pattern_dict = matching_module.apply(class_pers_img, img)
            rep_err = matching_module.calculate_reprojection_error(compare_result_list, new_pattern_dict)
            if rep_err < 2 :
                return homo_result_image, new_pattern_dict
            else :
                return img, None
        else :
            return img, None
    else :
        return img, None

class Ransac_plane_fitting:
    """
    fitting plane using RANSAC
    """
    def __init__(self, points, max_iteration, distance_threshold):
        # self.points = []
        # for key in points:
        #     self.points.append(points[key])
        self.points = points
        self.max_iteration = max_iteration
        self.distance_threshold = distance_threshold

    def apply(self):
        inliers_result = []
        result_plane = []
        while self.max_iteration:
            self.max_iteration -=1
            random.seed()
            inliers=[]
            while len(inliers) < 3:
                random_index = random.randint(0, len(self.points)-1)
                if random_index in inliers:
                    continue
                else :
                    inliers.append(random_index)
            print(inliers[0],inliers[1], inliers[2])
            [x1, y1, z1] = self.points[inliers[0]]
            [x2, y2, z2] = self.points[inliers[1]]
            [x3, y3, z3] = self.points[inliers[2]]
            
            a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
            b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
            c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            d = -(a * x1 + b * y1 + c * z1)
            plane_length = max(0.1, math.sqrt(a*a + b*b + c*c))

            for n, point in enumerate(self.points):
                if n in inliers:
                    continue
                else :
                    [x, y, z] = point
                    distance = math.fabs(a*x + b*y + c*z +d)
                    if distance <= self.distance_threshold:
                        inliers.append(n)
                    
            if len(inliers) > len(inliers_result):
                inliers_result.clear()
                inliers_result = inliers
                result_plane = [a, b, c, d]
            
            print(f'*****{self.max_iteration}, {len(inliers)}*****')
            print(inliers_result)
            
        if len(inliers_result) < 10 :
            return [False, 0, 0, 0]
        else :
            return result_plane

    def apply_mean(self):
        inliers_result = []
        result_plane = [0, 0, 0, 0]
        count = 0
        while self.max_iteration:
            self.max_iteration -=1
            random.seed()
            inliers=[]
            while len(inliers) < 3:
                random_index = random.randint(0, len(self.points)-1)
                if random_index in inliers:
                    continue
                else :
                    inliers.append(random_index)
            print(inliers[0],inliers[1], inliers[2])
            [x1, y1, z1] = self.points[inliers[0]]
            [x2, y2, z2] = self.points[inliers[1]]
            [x3, y3, z3] = self.points[inliers[2]]
            
            a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
            b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
            c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            d = -(a * x1 + b * y1 + c * z1)
            if d < 500:    
                norm = math.sqrt(a*a + b*b + c*c)
                result_plane[0] += (a/norm)
                result_plane[1] += (b/norm)
                result_plane[2] += (c/norm)
                result_plane[3] += (d/norm)
                print(self.max_iteration, a, b, c, d)
                count+=1

        #cv.waitKey(0)       
        result_plane[0] /= count
        result_plane[1] /= count
        result_plane[2] /= count
        result_plane[3] /= count
           
        return result_plane
    
    def apply_combination(self):
        combination = [
            [4, 1, 10],
            [4, 2, 9],
            [4, 3, 7],
            [4, 5, 10],
            [4, 6, 10],
            [1, 7, 9],
            [81, 8, 4],
            [82, 10, 7]
            ]
        result_plane = [0, 0, 0, 0]
        for com in combination:
            [x1, y1, z1] = self.points[com[0]]
            [x2, y2, z2] = self.points[com[1]]
            [x3, y3, z3] = self.points[com[2]]

            vec12 = [x2 - x1, y2 - y1, z2 - z1]
            vec13 = [x3 - x1, y3 - y1, z3 - z1]

            
            a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
            b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
            c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

            norm = math.sqrt(a*a + b*b + c*c)
            result_plane[0] += (a/norm)
            result_plane[1] += (b/norm)
            result_plane[2] += (c/norm)
        
        result_plane[0] /= len(combination)
        result_plane[1] /= len(combination)
        result_plane[2] /= len(combination)
        result_plane[3] /= len(combination)

           
        return result_plane




def main():
    
    zed_cam, zed_mat, zed_runtime = zed_open()
    
    #File Path
    seg_model_path = 'model_0623_segmentation.engine'
    cls_model_path = 'wresnet.engine'
    class_image_path = '5th_pattern/5th_pattern.png'

    #Load TensorRT Model (Segmentation, Classification)
    seg_module = Segmentation(seg_model_path)
    cls_module = Classification(cls_model_path)
    matching_module = Matching(class_image_path)

    plt.ion()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    

    while True:
        left_input_ori = zed_image_load(zed_cam, zed_mat, zed_runtime, 'Left')
        left_result_img, left_result_dict = eve_button_2D_detection(seg_module, cls_module, matching_module, left_input_ori)

        right_input_ori = zed_image_load(zed_cam, zed_mat, zed_runtime, 'Right')
        right_result_img, right_result_dict = eve_button_2D_detection(seg_module, cls_module, matching_module, right_input_ori)
        if left_result_dict is not None:
            if right_result_dict is not None:
                xaxis_dict = {}
                yaxis_dict = {}
                depth_dict = {}
                points_3d_dict = {}
                focal_length = 700
                baseline = 6.3
                cx = 640
                cy = 360
                yaw = []
                for key in left_result_dict:
                    #if key != "Panel":
                    if True:
                        [x_l, y_l] = left_result_dict[key]
                        [x_r, y_r] = right_result_dict[key]
                        disparity = x_l - x_r 
                        pz = (focal_length * baseline) / disparity
                        px = (x_l - cx) * pz / focal_length
                        py = (y_l - cy) * pz / focal_length
                        points_3d_dict[key] = [px, py, pz]
                        text=str('%.1f, %.1f, %.1f' % (px, py, pz))
                        font=cv.FONT_HERSHEY_SIMPLEX
                        org_left = (x_l-5, y_l-15)
                        org_right = (x_r-5, y_r-15)
                        cv.putText(left_result_img, text, org_left,font, 0.4,(0,0,0),2)
                        cv.putText(right_result_img, text, org_right,font, 0.4,(0,0,0),2)
                #Calculate Plane using RANSAC fitting
                ran_plane_fit = Ransac_plane_fitting(points_3d_dict, 100, 0.1)
                #[vec_a, vec_b, vec_c, vec_d] = ran_plane_fit.apply()
                [vec_a, vec_b, vec_c, vec_d] = ran_plane_fit.apply_combination()
                if not vec_a:
                    print('not matching')
                else :
                    vec_norm = math.sqrt(vec_a * vec_a + vec_b * vec_b + vec_c * vec_c)

                    if vec_c < 0:
                        [tx, ty, tz, tu, tv, tw] = [points_3d_dict['Panel'][0], points_3d_dict['Panel'][1], points_3d_dict['Panel'][2], vec_a/vec_norm * 10, vec_b/vec_norm * 10,vec_c/vec_norm * 10]
                    else :
                        [tx, ty, tz, tu, tv, tw] = [points_3d_dict['Panel'][0], points_3d_dict['Panel'][1], points_3d_dict['Panel'][2], -vec_a/vec_norm * 10, -vec_b/vec_norm * 10,-vec_c/vec_norm * 10]
                    
                    x_test_points = []
                    y_test_points = []
                    z_test_points = []
                    for key in points_3d_dict:
                        x_test_points.append(points_3d_dict[key][0])
                        y_test_points.append(points_3d_dict[key][1])
                        z_test_points.append(points_3d_dict[key][2])
                    ax.clear()
                    ax.set_xlim3d(-30, 30)
                    ax.set_ylim3d(-30, 30)
                    ax.set_zlim3d(10, 50)
                    #ax.view_init(-90, -90)
                    ax.quiver(tx, ty, tz, tu, tv, tw)
                    ax.scatter(np.array(x_test_points),np.array(y_test_points), np.array(z_test_points),c= 'red', marker ='o', s=15, cmap ='Greens')
                    
                    plt.show()
                    plt.pause(0.0001)


                #Calculate Yaw
                for pp_list in matching_module.pattern_pair:
                    [px1, py1, pz1] = points_3d_dict[pp_list[0]]
                    [px2, py2, pz2] = points_3d_dict[pp_list[1]]
                    dx = px2 - px1
                    dz = pz2 - pz1
                    print(pp_list)
                    print('dx : ', dx)
                    print('dz : ', dz)
                    yaw.append(math.atan2(dz, dx) * 180 / math.pi)
                print(yaw)
                average_yaw = sum(yaw)/ len(yaw)
                print(average_yaw)
                text=str('angle : %2.f' % average_yaw)
                font=cv.FONT_HERSHEY_SIMPLEX
                org_l = (left_result_dict['Panel'][0] + 100, left_result_dict['Panel'][1] + 0)
                org_r = (right_result_dict['Panel'][0] + 100, right_result_dict['Panel'][1] + 0)
                cv.putText(left_result_img, text, org_l,font, 1,(255,0,0),2)
                cv.putText(right_result_img, text, org_r,font, 1,(255,0,0),2)

                cv.imshow('Left', left_result_img)
                cv.imshow('Right', right_result_img)
                #cv.waitKey(0)
    
        if cv.waitKey(1)& 0xFF == ord('q'):
            break

if __name__ =='__main__':
  main()
