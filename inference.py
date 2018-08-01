# -*- coding:utf-8 -*-
import os
import sys
import argparse
import json
import numpy as np
import multiprocessing
import urllib
import caffe
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import time
# import yaml

from process import Producer_Of_ImageNameQueue, Producer_Of_ImageDataQueue_And_consumer_Of_imageNameQueue, Consumer_Of_ImageDataQueue_Inference

from refindet import Model


def getFilePath_FileNameNotIncludePostfix(fileName=None):
    justFileName = os.path.split(fileName)[-1]
    filePath = os.path.split(fileName)[0]
    if '.' in justFileName:
        justFileName = justFileName[:justFileName.rfind('.')]
    return [filePath, justFileName, os.path.join(filePath, justFileName)]


def initModels(modelFile=None, gpuId=None, inputFileName=None, beginIndex=None):
    """
    """
    model_list = []
    with open(modelFile, 'r') as m_f:
        line = m_f.readlines()
        modelParams = json.loads(line)
    
    saveFileName_model1 = getFilePath_FileNameNotIncludePostfix(
        fileName=inputFileName)[-1]+'_'+str(beginIndex)+"-result.json"
    saveFileName_model2 = getFilePath_FileNameNotIncludePostfix(
        fileName=inputFileName)[-1]+'_'+str(beginIndex)+"-resultModel2.json"
    
    model_1 = modelParams[0]
    model_1['gpuId'] = gpuId
    model_1['saveResultFileName'] = saveFileName_model1
    model_1['imageSize'] = 320
    
    model_2 = modelParams[1]
    model_2['gpuId'] = gpuId
    model_2['saveResultFileName'] = saveFileName_model2
    model_2['imageSize'] = 320
    
    m1=Model(model_1)
    m2=Model(model_2)

    model_list.append(m1)
    model_list.append(m2)

    return model_list

def mainProcessFun(param_dict_JsonStr=None):
    param_dict = json.loads(param_dict_JsonStr)
    countOfgetUrlDataThread = param_dict['imageDataProducerCount']
    modelList = initModels(modelFile=param_dict['modelFile'], gpuId=param_dict['gpuId'],
                           inputFileName=param_dict['inputFileName'], beginIndex=param_dict['beginIndex'])

    print("main process begin running")
    imageNameQueue = multiprocessing.Queue()
    imageName_lock = multiprocessing.Lock()
    imageDataQueue = multiprocessing.Queue()
    imageData_lock = multiprocessing.Lock()
    producer_Of_ImageNameQueue = Producer_Of_ImageNameQueue(
        imageNameQueue, imageName_lock, param_dict_JsonStr, "producer_Of_ImageNameQueue-"+str(1))
    producer_Of_ImageNameQueue.daemon = True
    producer_Of_ImageNameQueue.start()
    time.sleep(10)
    threadList = []
    for i in range(1, countOfgetUrlDataThread+1):
        threadName = "producer_Of_ImageDataQue_And_consumer_Of_imageNameQueue-" + \
            str(i)
        produce_and_consumer = Producer_Of_ImageDataQueue_And_consumer_Of_imageNameQueue(
            imageNameQueue, imageName_lock, imageDataQueue, imageData_lock, param_dict_JsonStr, threadName)
        threadList.append(produce_and_consumer)

    consumer_inference = Consumer_Of_ImageDataQueue_Inference(
        imageDataQueue, imageData_lock, param_dict_JsonStr, "consumer_inference-"+str(1), modelList)

    for i_thread in threadList:
        i_thread.daemon = True
        i_thread.start()
    consumer_inference.start()
    time.sleep(10)
    producer_Of_ImageNameQueue.join()
    for i_thread in threadList:
        i_thread.join()
        # eval('produce_and_consumer-{}.join()'.format(i))
    consumer_inference.join()
    print("main process end")
    pass


def parser_args():
    parser = argparse.ArgumentParser('bk detect caffe  refineDet model!')
    # url list file name
    parser.add_argument('--urlfileName', dest='urlfileName', help='url file name',
                        default=None, type=str, required=True)
    parser.add_argument('--urlfileName_beginIndex', dest='urlfileName_beginIndex', help='begin index in the url file name',
                        default=0, type=int)
    parser.add_argument('--gpu_id', dest='gpu_id', help='The GPU ide to be used',
                        default=0, type=int)
    parser.add_argument('--modelFile', required=True, dest='modelFile', help='models params file',
                        default=None, type=str)
    return parser.parse_args()

args = parser_args()

def main():
    param_dict = dict()
    param_dict['inputFileName'] = args.urlfileName
    param_dict['beginIndex'] = args.urlfileName_beginIndex
    param_dict['gpuId'] = int(args.gpu_id)
    param_dict['modelFile'] = args.modelFile
    param_dict['imageDataProducerCount'] = 2
    param_dict['urlFlag'] = True
    param_dict_JsonStr = json.dumps(param_dict)
    print(param_dict)
    mainProcessFun(param_dict_JsonStr=param_dict_JsonStr)
    pass

if __name__ == '__main__':
    main()
