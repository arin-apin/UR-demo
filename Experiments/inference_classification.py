from tflite_runtime.interpreter import Interpreter
import numpy as np
import tensorflow as tf
from PIL import Image as ImagePIL
import time 
from numpy import asarray, transpose, tri
from PIL import ImageTk
import tflite_runtime.interpreter as tflite
import pprint
import cv2

pp = pprint.PrettyPrinter(indent=4) # Set Pretty Print Indentation
model = "/path/to/.tflite"
labels_path = "/path/to/labels.txt"
#img = "/path/to/image"


model = "/home/gemma/Arinapin/Speech recognition/gestos2/clients_Fd6ROLFygUROdCkBbME0KJNmUAG3_L29Fg0jIi5bMpRMy9fwX_lD8eTCmzjR40dXFcbock_lD8eTCmzjR40dXFcbock.tflite"
labels_path = "/home/gemma/Arinapin/Speech recognition/gestos2/labels.txt"
#img = "/home/gemma/Arinapin/Speech recognition/cero.jpg"




#load labels from ffile. 
def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

labels = load_labels(labels_path)

#Model
def perform_inference(img):

    global interpreter, input_details, output_details
    interpreter = tflite.Interpreter(model)
    interpreter.allocate_tensors()
        #experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    # print('tensor input', input_details)
    global size
    size = [width, height]
    print(size)

    inicio=time.time()
    #Open the image
    img = cv2.imread(img)
    img_resized = cv2.resize(img, (width, height))

    #method 2
    # img_infe = ImagePIL.open(img)
    # img_resized= img_infe.convert('RGB').resize(size, ImagePIL.ANTIALIAS)
    input_data = np.array(asarray(img_resized), dtype=np.float32)
    input_data = np.expand_dims(input_data , axis=0)

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    print(input_details[0])
    print(floating_model)
    #floating_model = input_details[0]['dtype'] == np.float32


    print("-> Input data:", input_data)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    tensor_result= interpreter.get_tensor(output_details[0]['index'])[0]

    print(tensor_result)
    results = np.squeeze(tensor_result)
    #[::-1] is a trick in python to obtain a list in the opposite order. 
    top_k= results.argsort()[-5:][::-1]

    #We get the top 5 results 
    result=''
    #print(top_k)
    for i in top_k:
        result+=('{:08.6f}: {}'.format(float(tensor_result[i]), labels[i]))+"\n"
    result=result+"Inference time: "+str(time.time()-inicio)
    #print(resultado, '\n')
    resultado_max= result.partition('\n')[0]+'\n'+ result.split('\n')[-1] 

    fin = time.time()
    print("Inference time:" , fin-inicio)
    print(resultado_max)
    print("\n",result)

    return resultado_max

