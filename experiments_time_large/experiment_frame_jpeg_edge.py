import sys
# setting path
sys.path.append('../')

import cherrypy
import json

import cv2
from  pytorchyolo import detect, models_split_large
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
import torchvision.transforms as transforms
import numpy as np
from pytorchyolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
import pandas as pd
import time
import torch
import pickle
from split_framework.yolov3_tensor_jpeg import SplitFramework
from torch.profiler import profile, record_function, ProfilerActivity
import simplejpeg

class TailModelService:
    exposed = True

    @cherrypy.tools.accept(media='text/plain')
    def __init__(self, split_layer,dummy_tensor) -> None:
        self.model = models_split_large.load_model(
            "../pytorchyolo/config/yolov3.cfg",
            "../pytorchyolo/weights/yolov3.weights")
        self.model.set_split_layer(split_layer) # layer <7
        self.model = self.model.eval()
        self.dummy_tensor = dummy_tensor
        self.sf = SplitFramework(device="cuda")
        self.sf.set_reference_tensor(dummy_tensor)
        self.time_start = torch.cuda.Event(enable_timing=True)
        self.time_end = torch.cuda.Event(enable_timing=True)
    
    def convert_rgb_frame_to_tensor(self, image):
        img_size = 416
        # Configure input
        input_img = transforms.Compose([
        DEFAULT_TRANSFORMS,
        Resize(img_size)])(
            (image, np.zeros((1, 5))))[0].unsqueeze(0)
        input_img = input_img.cuda()

        return input_img

    def POST(self, *uri):
        urilen = len(uri)
        if urilen != 0 :
            print(uri[0])
        if uri[0] == "frame_jpeg":
            body = cherrypy.request.body.read()
            data = pickle.loads(body)
            print("Processing: ",data["id"])
            decode_time = 0
            model_time = 0
            ################## Perform Object detection #############################
            with torch.no_grad():
                ######## Framework decode ##########
                self.time_start.record()
                frame = simplejpeg.decode_jpeg(data["frame"])
                frame_tensor = self.convert_rgb_frame_to_tensor(frame)
                self.time_end.record()
                torch.cuda.synchronize()
                decode_time = self.time_start.elapsed_time(self.time_end)
                
                ######## Framework decode ##########
                self.time_start.record()
                head_output = self.model(frame_tensor,1)
                inference_result = self.model(head_output,2)
                detection = non_max_suppression(inference_result, 0.5, 0.5)
                # print(detection)
                self.time_end.record()
                torch.cuda.synchronize()
                tail_time = self.time_start.elapsed_time(self.time_end)
            
            ##################### Collect resource usage ##########################
            test_restult = {"id":data["id"], 
                            "model_time": tail_time,
                            "decode_time":decode_time,
                            "detection":detection }
            return pickle.dumps(test_restult)



if __name__ == "__main__":
    split_layer = 4
    dummy_tensor = torch.rand([1, 64, 208, 208])
    tail_service = TailModelService(split_layer, dummy_tensor)

    conf = {
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.sessions.on': True,
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [('Content-Type', 'text/plain')]
        }
    }
    # set this address to host ip address to enable dockers to use REST api
    cherrypy.server.socket_host = "10.0.1.23"
    cherrypy.config.update(
        {'server.socket_port':8090 })

    # Blocking the terminal and show output, for debug
    cherrypy.tree.mount(tail_service, "/",conf)
    cherrypy.engine.start()
    cherrypy.engine.block()