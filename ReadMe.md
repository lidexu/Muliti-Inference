文件说明：
=======================
##文件说明
process.py: 多线程读写推理
refindet.py: refindet模型
inference.py: 多模型推理
inference传参说明：
inputFileName：图片url文件
beginIndex：起始url的index
gpuId ：推理时使用的gpuId
modelFile ： json文件，一个模型一个dict， dict结构如caffe模型参数所示

添加模型时修改ininModels()函数, 增加模型，同时在modelFile里提供相应模型文件路径

caffe模型文件参数：
{ 
    'model1':{
    'modelFileName' : modelFileName,
    'deployFileName' : deployFileName,
    'labelFileName' : labelFileName
    } 
}