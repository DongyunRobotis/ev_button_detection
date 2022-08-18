from multiprocessing.dummy import Process
import pycuda.driver as cuda
#import pycuda.autoinit
import tensorrt as trt
import numpy as np
from .image_proc import preprocess, visualize
import cv2 as cv

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)
    def __repr__(self):
        return self.__str__()

class Segmentation:
    def __init__(self, path):
        self.seg_model_engine, self.seg_model_inputs, self.seg_model_outputs, self.seg_model_bindings, self.seg_model_stream, self.seg_model_context, self.seg_model_cfx = trt_model_load(path)
    
    def infer(self, img):
        self.input_image = img.copy()
        self.ori_h, self.ori_w, self.ori_c = img.shape
        self.inf_result = trt_segmentation(img, self.seg_model_inputs, self.seg_model_context, self.seg_model_bindings, self.seg_model_outputs, self.seg_model_stream)
        return self.inf_result

    def get_button_binary_image(self):
        self.btn_image = self.inf_result.copy()
        self.btn_image[np.where(self.btn_image == 2)] = 255
        self.btn_image[np.where(self.btn_image == 3)] = 255
        #self.btn_image[np.where(self.btn_image == 4)] = 255
        self.btn_image[np.where(self.btn_image == 5)] = 255
        #self.btn_image[np.where(self.btn_image == 6)] = 255
        self.btn_image = cv.resize(self.btn_image,(self.ori_w, self.ori_h))
        return self.btn_image
    
    def crop_patch(self):
        seg_patch_image = []
        seg_bbox = []
        bbox = []
        center_pts = []
        ret, thresh = cv.threshold(self.btn_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        kernel = np.ones((3,3),np.uint8)
        closing_img = cv.morphologyEx(thresh,cv.MORPH_CLOSE,kernel,iterations=2)
        contours, hierarchy = cv.findContours(closing_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        dst =  np.zeros_like(closing_img)
        for cnt in contours:
            hull = cv.convexHull(cnt)
            x, y, w, h = cv.boundingRect(cnt)
            if w < closing_img.shape[1] - 20 and w > 10:
                M = cv.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                center_pts.append([cx, cy])
                bbox.append([x, y, x + w, y + h])
                cv.drawContours(dst, [hull], 0, 255, -1)
        self.input_image[np.where(dst != 255)] = 0
        seg_bbox = set([tuple(bbx) for bbx in bbox])
        for bb in seg_bbox:
            crop_img = self.input_image[bb[1] : bb[3], bb[0] : bb[2]]
            seg_patch_image.append(crop_img)

        return seg_patch_image, seg_bbox, center_pts
    
    def result_image(self):
        result_color = visualize(self.input_image, self.inf_result)
        result_color = cv.resize(result_color,(self.ori_w, self.ori_h))
        return result_color

class Classification:
    def __init__(self, path):
        self.cls_model_engine, self.cls_model_inputs, self.cls_model_outputs, self.cls_model_bindings, self.cls_model_stream, self.cls_model_context, self.cls_model_cfx = trt_model_load(path)
        
    def infer(self, patch_image, patch_bbox, patch_center_pts):
        class_bbox = []
        class_patch_center_pts = []
        list_seg_patch_bbox = list(patch_bbox)
        list_seg_patch_center_pts = list(patch_center_pts)
        for n, patch_image in enumerate(patch_image):
            inf_class = trt_classification(patch_image, self.cls_model_inputs, self.cls_model_context, self.cls_model_bindings, self.cls_model_outputs, self.cls_model_stream)
            class_bbox.append([inf_class, list_seg_patch_bbox[n][0], list_seg_patch_bbox[n][1], list_seg_patch_bbox[n][2], list_seg_patch_bbox[n][3]])
            class_patch_center_pts.append([inf_class, list_seg_patch_center_pts[n][0], list_seg_patch_center_pts[n][1]])
            #class_patch_center_pts.append([inf_class, (list_seg_patch_bbox[n][0] + list_seg_patch_bbox[n][2]) * 0.5, (list_seg_patch_bbox[n][1] + list_seg_patch_bbox[n][3]) * 0.5])

        return [class_bbox, class_patch_center_pts]

    def make_class_image(self, input, cls_bbx):
        output = input.copy()
        for cb in cls_bbx:
            cv.circle(output, ((int)((cb[1] + cb[3]) * 0.5), (int)((cb[2] + cb[4]) * 0.5)), 3, (255, 0, 0), -1)
            text=str(cb[0])
            font=cv.FONT_HERSHEY_SIMPLEX
            org = (cb[1], cb[2]-15)
            cv.putText(output, text, org, font, 0.5,(0,0,255),2)
        return output

def do_inference(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    return [out.host for out in outputs]

def trt_model_load(engine_path):
    cuda.init()
    cfx = cuda.Device(0).make_context()
    tensorrt_file_name = engine_path
    TRT_LOGGER = trt.Logger()
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    trt_runtime = trt.Runtime(TRT_LOGGER)
    
    with open(tensorrt_file_name, 'rb') as f:
        engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
    inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    context = engine.create_execution_context()
    context.profiler = trt.Profiler()
    return engine, inputs, outputs, bindings, stream, context, cfx

def trt_segmentation(img_path, trt_input, trt_context, trt_binding, trt_output, trt_stream):
    input_data = preprocess(img_path, 512, 512, False)
    input_data = np.array(input_data, dtype=np.float32, order='C')
    trt_input[0].host = input_data
    trt_outputs = do_inference(context=trt_context, bindings=trt_binding, inputs=trt_input, outputs=trt_output, stream=trt_stream)
    trt_outputs = trt_outputs[0].reshape(512, 512, 1)
    inf_image = np.array(trt_outputs, dtype=np.uint8)
    return inf_image

def trt_classification(input_image, trt_input, trt_context, trt_binding, trt_output, trt_stream):
    input_data = preprocess(input_image,54, 54, True)
    input_data = np.array(input_data, dtype=np.float32, order='C')
    trt_input[0].host = input_data
    ll, dl1, dl2, dl3, dl4, dl5 = do_inference(context=trt_context, bindings=trt_binding, inputs=trt_input, outputs=trt_output, stream=trt_stream)
    if dl2.argmax() == 10:
      return dl1.argmax()
    else :
        return dl1.argmax() * 10 + dl2.argmax()

