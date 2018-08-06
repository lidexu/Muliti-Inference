import os
import sys
import cv2
import json
import caffe
import numpy as np

from google.protobuf import text_format
from caffe.proto import caffe_pb2
import time

class Model(object):
    def __init__(self, paramDict):
        self.paramDict=paramDict
        # self.net = None
        self.label_list = None
        # self.init_Net()
        # self.init_Model()
        
    
    def init_Net(self):
        self.saveFile = open(self.paramDict['saveResultFileName'], 'w')
        self.gpuId = int(self.paramDict['gpuId'])
        self.modelFileName = self.paramDict['modelFileName']
        self.deployFileName =  self.paramDict['deployFileName']
        self.labelFileName = self.paramDict['labelFileName']
        self.imageSize = self.paramDict['imageSize']
    
    def init_Model(self):
        caffe.set_mode_gpu()
        caffe.set_device(self.gpuId)
        self.net = caffe.Net(str(self.deployFileName),
        str(self.modelFileName), caffe.TEST)
        with open(str(self.labelFileName), 'r') as f:
            self.label_list = caffe_pb2.LabelMap()
            text_format.Merge(str(f.read()), self.label_list)

    def preProcess(self, oriImage=None):
        img = cv2.resize(oriImage, (self.imageSize, self.imageSize))
        img = img.astype(np.float32, copy=False)
        img = img - np.array([[[103.52, 116.28, 123.675]]])
        img = img * 0.017
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))
        return img
    
    def postProcess(self, output=None, imagePath=None, height=None, width=None):

        w = width
        h = height
        bbox = output[0, :, 3:7] * np.array([w, h, w, h])
        clas = output[0, :, 1]
        conf = output[0, :, 2]
        result_dict = dict()
        result_dict['bbox'] = bbox.tolist()
        result_dict['cls'] = clas.tolist()
        result_dict['conf'] = conf.tolist()
        result_dict['imagePath'] = imagePath
        self.saveFile.write(json.dumps(result_dict) + '\n')
        self.saveFile.flush()

    def inference(self, oriImgData=None, imagePath=None):
        imgDataHeight = oriImgData.shape[0]
        imgDataWidth = oriImgData.shape[1]
        imgData = self.preProcess(oriImgData)
        self.net.blobs['data'].data[...] = imgData
        output = self.net.forward()
        self.postProcess(output=output['detection_out'][0], imagePath=imagePath,
                        height=imgDataHeight, width=imgDataWidth )