import multiprocessing
import time
import json
import urllib
import cv2
import numpy as np
from refindet import Model


class Producer_Of_ImageNameQueue(multiprocessing.Process):

    def __init__(self, imageNameQueue, paramDictJsonStr, threadName, imageName_lock):
        multiprocessing.Process.__init__(self)
        self.imageNameQueue = imageNameQueue
        self.paramDict = json.loads(paramDictJsonStr)
        self.threadName = threadName
        self.lock = imageName_lock

    def getTimeFlag(self):
        return time.strftime("%Y:%m:%d:%H:%M:%S", time.localtime())

    def run(self):
        print("LOGINFO---%s---Thread %s begin running" %(self.getTimeFlag(), self.threadName))
        fileName = self.paramDict['inputFileName']
        beginIndex = int(self.paramDict['beginIndex'])
        with open(fileName, 'r') as f:
            for line in f.readlines()[beginIndex:]:
                line = line.strip()
                if len(line) <= 0:
                    continue
                self.lock.acquire()
                try:
                    self.imageNameQueue.put(line)
                except:
                    print("LOGINFO---%s---Thread %s put image name in queue exception" %
                          (self.getTimeFlag(), self.threadName))
                finally:
                    self.lock.release()
        for i in range(int(self.paramDict['imageDataProducerCount'])):
            self.lock.acquire()
            self.imageNameQueue.put(None)
            self.lock.release()
        print("LOGINFO---%s---Thread %s end" %
              (self.getTimeFlag(), self.threadName))
        pass

class Producer_Of_ImageDataQueue_And_consumer_Of_imageNameQueue(multiprocessing.Process):
    def __init__(self, imageNameQueue, imageName_lock, imageDataQueue, imageData_lock, paramDictJsonStr, threadName):
        multiprocessing.Process.__init__(self)
        self.imageNameQueue = imageNameQueue
        self.imageDataQueue = imageDataQueue
        self.paramDict = json.loads(paramDictJsonStr)
        self.urlFlag = True
        self.threadName = threadName
        self.imageName_lock = imageName_lock
        self.imageData_lock = imageData_lock

    def getTimeFlag(self):
        return time.strftime("%Y:%m:%d:%H:%M:%S", time.localtime())

    def readImage_fun(self, isUrlFlag=None, imagePath=None):
        """
            isUrlFlag == True , then read image from url
            isUrlFlag == False , then read image from local path
        """
        im = None
        if isUrlFlag == True:
            try:
                data = urllib.urlopen(imagePath.strip()).read()
                nparr = np.fromstring(data, np.uint8)
                if nparr.shape[0] < 1:
                    im = None
            except:
                im = None
            else:
                try:
                    im = cv2.imdecode(nparr, 1)
                except:
                    im = None
            finally:
                return im
        else:
            im = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        if np.shape(im) == ():
            return None
        return im

    def run(self):
        print("LOGINFO---%s---Thread %s begin running" %
              (self.getTimeFlag(), self.threadName))
        self.urlFlag = self.paramDict['urlFlag']
        timeout_count = 0
        while True:
            try:
                self.imageName_lock.acquire()
                imagePath = self.imageNameQueue.get(block=True, timeout=60)
            except:
                self.imageName_lock.release()
                print("%s : %s  get timeout" %
                      (self.getTimeFlag(), self.threadName))
                time.sleep(3)
                timeout_count += 1
                if timeout_count > 5:
                    print("LOGINFO---%s---Thread exception,so kill %s" %
                          (self.getTimeFlag(), self.threadName))
                    break
                else:
                    time.sleep(10)
                    continue
            else:
                self.imageName_lock.release()
                if imagePath == None:
                    print("LOGINFO---%s---Thread %s Exiting" %
                          (self.getTimeFlag(), self.threadName))
                    break
                imgData = self.readImage_fun(
                    isUrlFlag=self.urlFlag, imagePath=imagePath)
                if np.shape(imgData) == () or len(np.shape(imgData)) != 3 or np.shape(imgData)[-1] != 3:
                    print("WARNING---%s---imagePath %s can't read" %
                          (self.getTimeFlag(), imagePath))
                else:
                    self.imageData_lock.acquire()
                    self.imageDataQueue.put([imagePath, imgData])
                    self.imageData_lock.release()
        self.imageData_lock.acquire()
        self.imageDataQueue.put(None)
        self.imageData_lock.release()
        print("LOGINFO---%s---Thread %s end" %
              (self.getTimeFlag(), self.threadName))
    pass

class Consumer_Of_ImageDataQueue_Inference(multiprocessing.Process):
    def __init__(self, imageDataQueue, imageData_lock, paramDictJsonStr, threadName, modelList):
        multiprocessing.Process.__init__(self)
        self.imageDataQueue = imageDataQueue
        self.paramDict = json.loads(paramDictJsonStr)
        self.threadName = threadName
        self.imageData_lock = imageData_lock
        self.modelList =  modelList
    
    def getTimeFlag(self):
        return time.strftime("%Y:%m:%d:%H:%M:%S", time.localtime())

    def run(self):
        print("LOGINFO---%s---Thread %s begin running" %
              (self.getTimeFlag(), self.threadName))
        endGetImageDataThreadCount = 0
        time_out_count = 0
        while True:
            try:
                self.imageData_lock.acquire()
                next_imageData = self.imageDataQueue.get(block=True, timeout=60)
            except:
                self.imageData_lock.release()
                print("%s  get timeout" % (self.threadName))
                time.sleep(3)
                time_out_count += 1
                if endGetImageDataThreadCount >= self.paramDict['imageDataProducerCount'] or time_out_count > 8:
                    print("LOGINFO---%s---Thread Exception so kill  %s " %
                          (self.getTimeFlag(), self.threadName))
                    break
                else:
                    time.sleep(10)
                    continue
            else:
                self.imageData_lock.release()
                if next_imageData == None:
                    endGetImageDataThreadCount += 1
                    if endGetImageDataThreadCount >= self.paramDict['imageDataProducerCount']:
                        print("LOGINFO---%s---Thread %s Exiting" %
                              (self.getTimeFlag(), self.threadName))
                        break
                else:
                    imagePath = next_imageData[0]
                    orginalImgData = next_imageData[1]
                    # self.test_model(orginalImgData=orginalImgData,imagePath=imagePath)
                    for model in self.modelList:
                        model.inference(oriImgData=orginalImgData, imagePath=imagePath)

        print("LOGINFO---%s---Thread %s end" %(self.getTimeFlag(), self.threadName))
    pass
