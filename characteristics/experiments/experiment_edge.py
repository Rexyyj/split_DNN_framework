import sys
# setting path
sys.path.append('../../')

import cherrypy
import json

import cv2
from  pytorchyolo import detect, models_split_tiny
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
import torchvision.transforms as transforms
import numpy as np
from pytorchyolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
import pandas as pd
import time
import torch
import pickle
from split_framework.yolov3_tensor_jpeg_v2 import SplitFramework
# from split_framework.yolov3_tensor_regression_chara import SplitFramework
from torch.profiler import profile, record_function, ProfilerActivity

def get_dummy_tensor(split_layer):
    if split_layer==8:
        model_split_layer = 7
        dummy_head_tensor = torch.rand([1, 128, 26, 26])
    elif split_layer==7:
        model_split_layer = 6
        dummy_head_tensor = torch.rand([1, 128, 52, 52])
    elif split_layer==6:
        model_split_layer = 5
        dummy_head_tensor = torch.rand([1, 64, 52, 52])
    elif split_layer==5:
        model_split_layer = 4
        dummy_head_tensor = torch.rand([1, 64, 104, 104])
    elif split_layer==4:
        model_split_layer = 3
        dummy_head_tensor = torch.rand([1, 32, 104, 104])
    elif split_layer==3:
        model_split_layer = 2
        dummy_head_tensor = torch.rand([1, 32, 208, 208])
    elif split_layer==2:
        model_split_layer = 1
        dummy_head_tensor = torch.rand([1, 16, 208, 208])
    elif split_layer==1:
        model_split_layer = 0
        dummy_head_tensor = torch.rand([1, 16, 416, 416])
    return model_split_layer,dummy_head_tensor


class TailModelService:
    exposed = True

    @cherrypy.tools.accept(media='text/plain')
    def __init__(self, split_layer,dummy_tensor) -> None:
        self.model = models_split_tiny.load_model(
            "../../pytorchyolo/config/yolov3-tiny.cfg",
            "../../pytorchyolo/weights/yolov3-tiny.weights")
        self.model.set_split_layer(split_layer) 
        self.model = self.model.eval()
        self.dummy_tensor = dummy_tensor
        self.sf = SplitFramework(device="cuda")
        self.sf.set_reference_tensor(dummy_tensor)
        self.time_start = torch.cuda.Event(enable_timing=True)
        self.time_end = torch.cuda.Event(enable_timing=True)
    

    def POST(self, *uri):
        urilen = len(uri)
        if urilen != 0 :
            print(uri[0])
        if uri[0] == "set_layer":
            body = cherrypy.request.body.read()
            data = pickle.loads(body)
            layer = data["split_layer"]
            model_split_layer, dummy_head_tensor = get_dummy_tensor(layer)
            self.model.set_split_layer(model_split_layer) 
            self.dummy_tensor = dummy_head_tensor
            self.sf.set_reference_tensor(self.dummy_tensor)
            response = {"set_layer":True}
            return pickle.dumps(response)
        if uri[0] == "reset":
            self.sf.set_reference_tensor(self.dummy_tensor)
            response = {"reset_status":True}
            return pickle.dumps(response)
        if uri[0] == "tensor_jpeg":
            body = cherrypy.request.body.read()
            data = pickle.loads(body)
            print("Processing: ",data["id"])
            decode_time = 0
            tail_time = 0
            ################## Perform Object detection #############################
            with torch.no_grad():
                ######## Framework decode ##########
                self.time_start.record()
                reconstructed_head_tensor = self.sf.split_framework_decode(data)
                self.time_end.record()
                torch.cuda.synchronize()
                decode_time = self.time_start.elapsed_time(self.time_end)
                
                ######## Framework decode ##########
                self.time_start.record()
                inference_result = self.model(reconstructed_head_tensor,2)
                detection = non_max_suppression(inference_result, 0.5, 0.5)
                # print(detection)
                self.time_end.record()
                torch.cuda.synchronize()
                tail_time = self.time_start.elapsed_time(self.time_end)
                ######## Framework decode ##########
            test_restult = {"id":data["id"], 
                            "tail_time": tail_time,
                            "decode_time":decode_time,
                            "detection":detection }
            return pickle.dumps(test_restult)



if __name__ == "__main__":
    split_layer = int(sys.argv[1])

    model_split_layer, dummy_head_tensor = get_dummy_tensor(split_layer)
        
    tail_service = TailModelService(model_split_layer, dummy_head_tensor)

    conf = {
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.sessions.on': True,
            'tools.response_headers.on': True,
            'tools.response_headers.headers': [('Content-Type', 'text/plain')]
        }
    }
    # set this address to host ip address to enable dockers to use REST api
    cherrypy.server.socket_host = "10.0.1.34"
    cherrypy.config.update(
        {'server.socket_port':8090 })

    # Blocking the terminal and show output, for debug
    cherrypy.tree.mount(tail_service, "/",conf)
    cherrypy.engine.start()
    cherrypy.engine.block()