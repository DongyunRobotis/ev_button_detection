import rclpy
from rclpy.node import Node
import cv2 as cv
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from .tools.trt_proc import Segmentation, Classification
from .tools.image_proc import preprocess, crop_patch, normalize, get_color_map_list, visualize, img_check_show, divide_button_img, Matching
from .tools.plane_fitting import Ransac_plane_fitting
import os
import math
import numpy as np
from geometry_msgs.msg import Pose, PoseArray
from multiprocessing import Process
import time
import pycuda.driver as cuda
import tensorrt as trt
from multiprocessing import Process, Pipe

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()
      
      
def segmentation_process(conn):
    seg_model_name = 'models/model_0623_segmentation.engine'
    #seg_model_name = 'models/384_model.engine'
    seg_class = Segmentation(seg_model_name)
    seg_class.seg_model_cfx.push()
    while True:
        msg = conn.recv()
        print(msg)
        if msg == 'start':
            #inference
            
            # read image
            img_msg = conn.recv()
            # seg_class.seg_model_cfx.push()
            # inf_image = seg_class.infer(img_msg)
            # bin_image = seg_class.get_button_binary_image()
            # crop_list = seg_class.crop_patch()
            # conn.send(crop_list)
            # seg_class.seg_model_cfx.pop()
            conn.send('test')
            msg = ''
            
        elif msg == 'end' :
            print('process_end')
            break
        else :
            print('Waiting mainprocess cmd')
        time.sleep(0.001)
        
def classification_process(conn):
    #cls_model_name = 'models/SVHN_Best_model.engine'
    #cls_model_name = 'models/gray_model.engine'
    cls_model_name = 'models/RexNet_model.engine'
    cls_class = Classification(cls_model_name)
    while True:
        msg = conn.recv()
        print(msg)
        if msg == 'start':
            #inference
            input_list = conn.recv()
            cls_class.cls_model_cfx.push()
            cls_result = cls_class.infer(input_list[0],input_list[1], input_list[2])
            conn.send(cls_result)
            cls_class.cls_model_cfx.pop()
            msg = ''
            
        elif msg == 'end' :
            print('process_end')
            break
        else :
            print('Waiting mainprocess cmd')
        time.sleep(0.001)

def make_class_image(input, cls_bbx):
    output = input.copy()
    for cb in cls_bbx:
        cv.circle(output, ((int)((cb[1] + cb[3]) * 0.5), (int)((cb[2] + cb[4]) * 0.5)), 3, (255, 0, 0), -1)
        text=str(cb[0])
        font=cv.FONT_HERSHEY_SIMPLEX
        org = (cb[1], cb[2]-15)
        cv.putText(output, text, org, font, 0.5,(0,0,255),2)
    return output

def convert_zed2mat(mat):
    a, b, c, d = cv.split(mat)
    cv_mat = cv.merge((a, b, c))
    
    left_cv_mat = cv_mat[:,: (int)(cv_mat.shape[1]/2)]
    right_cv_mat = cv_mat[:, (int)(cv_mat.shape[1]/2) : ]
    
    return left_cv_mat, right_cv_mat
  
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
    inf_result = seg_module.infer(img)
    #Visualize Segmentation Result
    vis_seg_result = seg_module.result_image()
    #Binarize Button Class
    btn_image = seg_module.get_button_binary_image()
    seg_patch_image, seg_patch_bbox, seg_patch_center_pts = seg_module.crop_patch()
    #Classification
    class_bbox, class_patch_center_pts = cls_module.infer(seg_patch_image, seg_patch_bbox, seg_patch_center_pts)
    img = cls_module.make_class_image(img, class_bbox)
    #Match Seg_Points & Pattern Points
    compare_pattern_list, compare_result_list = matching_module.calculate_matching_pts(class_patch_center_pts)
    #Class Image Load
    class_pers_img = matching_module.make_pattern_image()
    if len(compare_pattern_list) > 3:
        if matching_module.calculate_homography_matrix(compare_pattern_list, compare_result_list):
            homo_result_image, new_pattern_dict = matching_module.apply(class_pers_img, img)
            rep_err = matching_module.calculate_reprojection_error(compare_result_list, new_pattern_dict)
            if rep_err < 1 :
                return homo_result_image, new_pattern_dict
            else :
                return img, None
        else :
            return img, None
    else :
        return img, None
      
def calculate_3d_pts(l_img, r_img, l_result_dict, r_result_dict):
  points_3d_dict= {}
  focal_length = 668.154541015625
  baseline = 6.296257782
  cx = 645.5487060546875
  cy = 370.5072021484375
  for key in l_result_dict:
    #if key != "Panel":
    if True:
      [x_l, y_l] = l_result_dict[key]
      [x_r, y_r] = r_result_dict[key]
      disparity = x_l - x_r 
      pz = (focal_length * baseline) / disparity
      px = (x_l - cx) * pz / focal_length
      py = (y_l - cy) * pz / focal_length
      
      #points_3d_dict[key] = [px, py, pz]
      
      points_3d_dict[key] = convert_coordinate_optframe2fram([px, py, pz])
      text=str('%.1f, %.1f, %.1f' % (points_3d_dict[key][0], points_3d_dict[key][1], points_3d_dict[key][2]))
      font=cv.FONT_HERSHEY_SIMPLEX
      org_left = (x_l-40, y_l-15)
      org_right = (x_r-40, y_r-15)
      cv.putText(l_img, text, org_left,font, 0.3,(0,0,0),2)
      cv.putText(r_img, text, org_right,font, 0.3,(0,0,0),2)
  return points_3d_dict, l_img, r_img

class ImageSubscriber(Node):
  def __init__(self):
    super().__init__('image_subscriber')
    self.subscription = self.create_subscription(
      Image, 
      '/zedm/zed_node/stereo/image_rect_color', 
      self.listener_callback, 
      1)
    self.subscription
    self.pose_publisher = self.create_publisher(PoseArray, 'button_3d_pose', 10)
    self.left_img_publisher = self.create_publisher(Image, 'left_image_result', 10)
    self.right_img_publisher = self.create_publisher(Image, 'right_image_result', 10)
    self.br = CvBridge()
    #Segmentation
    self.seg_process1_parent_conn, self.seg_process1_child_conn = Pipe()
    #self.seg_process2_parent_conn, self.seg_process2_child_conn = Pipe()
    self.seg_process1 = Process(target = segmentation_process, args=(self.seg_process1_child_conn,))
    #self.seg_process2 = Process(target = segmentation_process, args=(self.seg_process2_child_conn,))
    self.seg_process1.start()
    #self.seg_process2.start()
    self.file = open("eve_detection_log.txt","w")
    
    
    #Classification
    # self.cls_process_parent_conn = []
    # self.cls_process_child_conn = []
    # self.cls_process = []
    
    # for process_num in range(1):
    #   temp_parent_conn, temp_child_conn = Pipe()
    #   self.cls_process_parent_conn.append(temp_parent_conn)
    #   self.cls_process_child_conn.append(temp_child_conn)
    #   self.cls_process.append(Process(target = classification_process, args=(self.cls_process_child_conn[process_num],)))
    #   self.cls_process[process_num].start()
  
  
  def listener_callback(self, data):
    start_time = time.time()
    #self.get_logger().info('Receiving zed Image')
    current_frame = self.br.imgmsg_to_cv2(data)
    left_input_ori, right_input_ori = convert_zed2mat(current_frame)
    test_img = cv.vconcat([left_input_ori, right_input_ori])
    print('check_zed')
    #Segmentation Process 2
    self.seg_process1_parent_conn.send('start')
    self.seg_process1_parent_conn.send(test_img)
    #self.seg_process2_parent_conn.send('start')
    #self.seg_process2_parent_conn.send(right_input_ori)
    result1= self.seg_process1_parent_conn.recv()
    #result2= self.seg_process2_parent_conn.recv()
    #cv.imshow('l', result1)
    #cv.imshow('r', result2)
    
    #Classification Process 4
    # self.cls_process_parent_conn[0].send('start')
    # self.cls_process_parent_conn[0].send(result1)
    # #self.cls_process_parent_conn[1].send('start')
    # #self.cls_process_parent_conn[1].send(result2)
    # cls_result1 = self.cls_process_parent_conn[0].recv()
    # #cls_result2 = self.cls_process_parent_conn[1].recv()
    # left_img = make_class_image(test_img, cls_result1[0])
    # #right_img = make_class_image(right_input_ori, cls_result2[0])
    # cv.imshow('l', left_img)
    # #cv.imshow('r', right_img)
    #cv.waitKey(1)
    end_time = time.time()
    log = f'time : {end_time - start_time}s\n'
    self.file.write(log)
    # print(cls_result)
    
      
def main(args=None):
  rclpy.init(args=args)
  image_subscriber = ImageSubscriber()
  rclpy.spin(image_subscriber)
  image_subscriber.destroy_node()
  image_subscriber.file.close()
  rclpy.shutdown()
  
if __name__ == '__main__':
  
  main()
