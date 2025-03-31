import sys
# setting path
sys.path.append('../../')
sys.path.append('../')

import cherrypy

from  pytorchyolo import models_split_tiny
import torch
import pickle
from split_framework.split_framework_dynamic_direct import SplitFramework

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
            "../ckpt/yolov3_ckpt_300.pth")
        self.model.set_split_layer(split_layer) 
        self.model = self.model.eval()
        self.dummy_tensor = dummy_tensor
        self.sf = SplitFramework(device="cuda", model= self.model)
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
        if uri[0] == "tensor":
            body = cherrypy.request.body.read()
            data = pickle.loads(body)
        
            ################## Perform Object detection #############################
            response = self.sf.split_framework_service(data)
            return response



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
        {'server.socket_port':8092 })

    # Blocking the terminal and show output, for debug
    cherrypy.tree.mount(tail_service, "/",conf)
    cherrypy.engine.start()
    cherrypy.engine.block()