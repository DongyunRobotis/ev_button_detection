from tracemalloc import start
import rclpy
from rclpy.node import Node
import cv2 as cv
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from .tools.trt_proc import Segmentation, Classification
from .tools.image_proc import preprocess, normalize, get_color_map_list, visualize, img_check_show,  Matching
from .tools.plane_fitting import Plane_Fitting
import os
import math
import numpy as np
from geometry_msgs.msg import Pose, PoseArray
from multiprocessing import Process
import time
import pycuda.driver as cuda
import tensorrt as trt
from multiprocessing import Process, Pipe

def convert_zed2mat(mat):
    a, b, c, d = cv.split(mat)
    cv_mat = cv.merge((a, b, c))
    
    left_cv_mat = cv_mat[:,: (int)(cv_mat.shape[1]/2)]
    right_cv_mat = cv_mat[:, (int)(cv_mat.shape[1]/2) : ]
    
    vconcat_img = cv.vconcat([left_cv_mat, right_cv_mat])
    
    return vconcat_img
  
def convert_coordinate_optframe2fram(pts):
  convert_mat = [[0, 0, 1],[-1, 0, 0],[0, -1, 0]]
  tmp_pts_x = convert_mat[0][0] * pts[0] + convert_mat[0][1] * pts[1] + convert_mat[0][2] * pts[2]
  tmp_pts_y = convert_mat[1][0] * pts[0] + convert_mat[1][1] * pts[1] + convert_mat[1][2] * pts[2]
  tmp_pts_z = convert_mat[2][0] * pts[0] + convert_mat[2][1] * pts[1] + convert_mat[2][2] * pts[2]
  convert_pts = [tmp_pts_x, tmp_pts_y, tmp_pts_z]
  return convert_pts

def get_pose_array(pts_3d_dict, vec):
  result_pose = PoseArray()
  result_pose.header.frame_id = 'zedm_left_camera_frame'
  #result_pose.header.stamp = Node.get_clock().now().to_msg()
  for key in pts_3d_dict:
    tmp_pose = Pose()
    tmp_pose.position.x = pts_3d_dict[key][0] * 0.01
    tmp_pose.position.y = pts_3d_dict[key][1] * 0.01
    tmp_pose.position.z = pts_3d_dict[key][2] * 0.01
    tmp_pose.orientation = convert_vector2quat(vec)
    result_pose.poses.append(tmp_pose)
  return result_pose

def convert_vector2quat(vec3d):
  up_vector = [-1.0, 0.0, 0.0]
  d = math.sqrt(vec3d[0]**2 + vec3d[1]**2 + vec3d[2]**2)
  vec3d[0] /= d
  vec3d[1] /= d
  vec3d[2] /= d
  right_axis_vector = np.cross(up_vector, vec3d )
  theta = np.dot(vec3d, up_vector)
  angle = 1.0 * math.acos(theta)
  quat = Pose()
  quat.orientation.x = right_axis_vector[0] * math.sin(angle * 0.5) 
  quat.orientation.y = right_axis_vector[1] * math.sin(angle * 0.5)
  quat.orientation.z = right_axis_vector[2] * math.sin(angle * 0.5)
  quat.orientation.w = math.cos(angle * 0.5)
  
  return quat.orientation
  
def eve_button_2D_detection(seg_module, cls_module, matching_module, img):
    #TensorRT Segmentation Inference
    #re_img = cv.resize(img, (512, 512))
    inf_result = seg_module.infer(img)
    # rect_class = inf_result[4, :, :] + inf_result[3, :, :] + inf_result[5, :, :] + inf_result[6, :, :]
    
    # rect_class = rect_class * 255
    # rect_class = np.array(rect_class, dtype=np.uint8)
    # _, sure_bg = cv.threshold(rect_class, 127, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # _, sure_fg = cv.threshold(rect_class, 200, 255, cv.THRESH_BINARY)
    # kernel = np.ones((3,3),np.uint8)
    # sure_fg = cv.morphologyEx(sure_fg,cv.MORPH_OPEN,kernel,iterations=3)
    # sure_fg = np.uint8(sure_fg)
    # sure_bg = np.uint8(sure_bg)

    # unknown = cv.subtract(sure_bg, sure_fg)
    # ret, markers = cv.connectedComponents(sure_fg)
    # markers = markers + 1
    # markers[unknown == 255] = 0
    # markers = cv.watershed(re_img,markers)
    # re_img[markers == -1] = [255,0,0]
    # cv.imshow('res', sure_bg)
    # cv.imshow('rect_class', rect_class)
    #Visualize Segmentation Result
    vis_seg_result = seg_module.result_image()
    vis_seg_result = cv.resize(vis_seg_result, (768, 768))
    cv.imshow('vis_seg_result', vis_seg_result) 
    cv.waitKey(1)
    
    # seg_patch_image, seg_patch_center_pts = seg_module.crop_patch()
    return None, None
    #Classification
    # class_patch_center_pts = cls_module.infer(seg_patch_image, seg_patch_center_pts)
    # img = cls_module.make_class_image(seg_module.input_image, class_patch_center_pts)
    # #Matching
    # left_patch_cls_center_pts = []
    # right_patch_cls_center_pts = []
    # for cpcp in class_patch_center_pts:
    #   if cpcp[2] < 720:
    #     left_patch_cls_center_pts.append(cpcp)
    #   else :
    #     temp = [cpcp[0], cpcp[1], cpcp[2] - 720]
    #     right_patch_cls_center_pts.append(temp)
    # # #Match Seg_Points & Pattern Points
    
    # cv.imshow('vis_seg_result', img) 
    # cv.waitKey(1)
    # return calculate_restore_pts(left_patch_cls_center_pts, matching_module), calculate_restore_pts(right_patch_cls_center_pts, matching_module)
    
      
def calculate_restore_pts(patch_center_pts, match_module):
  compare_pattern_list, compare_matching_list = match_module.calculate_matching_pts(patch_center_pts)
  if len(compare_pattern_list) > 3:
      if match_module.calculate_homography_matrix(compare_pattern_list, compare_matching_list):
          new_pattern_dict = match_module.apply()
          rep_err = match_module.calculate_reprojection_error(compare_matching_list, new_pattern_dict)
          if rep_err < 2 :
              return new_pattern_dict
          else :
              return None
      else :
          return None
  else :
      return None
  
      
def calculate_3d_pts(l_result_dict, r_result_dict):
  points_3d_dict= {}
  focal_length = 668.154541015625
  baseline = 6.296257782
  cx = 645.5487060546875
  cy = 370.5072021484375
  for key in l_result_dict:
    if True:
      [x_l, y_l] = l_result_dict[key]
      [x_r, y_r] = r_result_dict[key]
      disparity = x_l - x_r 
      pz = (focal_length * baseline) / disparity
      px = (x_l - cx) * pz / focal_length
      py = (y_l - cy) * pz / focal_length
      points_3d_dict[key] = convert_coordinate_optframe2fram([px, py, pz])
  return points_3d_dict

class ImageSubscriber(Node):
  def __init__(self):
    super().__init__('image_subscriber')
    self.subscription = self.create_subscription(
      Image, 
      '/zedm/zed_node/stereo/image_rect_color', 
      self.listener_callback, 
      10)
    self.subscription
    self.pose_publisher = self.create_publisher(PoseArray, 'button_3d_pose', 10)
    self.br = CvBridge()
    #self.seg_module = Segmentation('models/model_0623_segmentation.engine')
    #self.seg_module = Segmentation('models/model_0816_with_softmax.engine')
    self.seg_module = Segmentation('models/model_0817_argmax.engine')
    
    self.cls_module = Classification('models/gray_model_0729.engine')
    self.matching_module = Matching('models/5th_pattern/5th_pattern.png')
    self.file = open("eve_detection_log.txt","w")
    self.toltal_time = 0
    self.time_count = 0
   
  def listener_callback(self, data):
    current_frame = self.br.imgmsg_to_cv2(data)
    zed_mat_vconcat = convert_zed2mat(current_frame)
    #try :
    if True:
      start_time = time.time()
      left_button_center_pts_2d_dict, right_button_center_pts_2d_dict = eve_button_2D_detection(self.seg_module, self.cls_module, self.matching_module, zed_mat_vconcat)
      
      if left_button_center_pts_2d_dict is not None:
        if right_button_center_pts_2d_dict is not None:
            button_center_pts_3d_dict = calculate_3d_pts(left_button_center_pts_2d_dict, right_button_center_pts_2d_dict)
            ran_plane_fit = Plane_Fitting(button_center_pts_3d_dict, 1000, 1)
            [vec_x, vec_y, vec_z, _] = ran_plane_fit.apply_combination()
            buttons_center_pose = get_pose_array(button_center_pts_3d_dict, [vec_x, vec_y, vec_z])
            self.pose_publisher.publish(buttons_center_pose)
            
      end_time = time.time()
      self.toltal_time += (end_time - start_time)
      self.time_count += 1
      log = f'\nAvg Time : {left_button_center_pts_2d_dict}s\n'
      self.file.write(log)
      
    # except :
    #   print('Zero Division Error')
    

  
def main(args=None):
  
  rclpy.init(args=args)
  image_subscriber = ImageSubscriber()
  rclpy.spin(image_subscriber)
  image_subscriber.destroy_node()
  image_subscriber.file.close()
  rclpy.shutdown()
  
if __name__ == '__main__':
  
  main()
