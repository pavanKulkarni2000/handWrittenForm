from tensorflow.keras.models import load_model
import cv2
import numpy as np

class WidgetClassifier:
  def __init__(self,path="./widget_models"):
    self.models={
      "radio":binaryClassifier3(path+"/radio.model"),
      "checkBox":binaryClassifier2(path+"/check_box.h5"),
      "table":binaryClassifier1(path+"/table.model"),
      "dropDown":binaryClassifier2(path+"/drop_down.h5"),
      "line":binaryClassifier1(path+"/line.model",(True,False)),
      "textBox":binaryClassifier2(path+"/text_box.h5"),
      "date":binaryClassifier2(path+"/date.h5")
    }
  def classify(self,img,ref):
    h,w=img.shape
    predictions=[]
    models_dict=self.models.copy()
    if w<1.5*ref:
      for w in ['line','date','table','textBox','dropDown']:
        models_dict.pop(w)
    else:
      for w in ['radio','checkBox']:
        models_dict.pop(w)
      if h>ref:
        models_dict.pop('line')
    for k,v in models_dict.items():
      p=v.predict(img)
      if p[0]:
        predictions.append((k,p[1]))
    if len(predictions) is 0:
      return None
    else:
      m= max(predictions,key=lambda x:x[1])
      if m[1]<0.75:
        return tuple([False])+m
      return tuple([True])+m

class binaryClassifier1:
  def __init__(self,path,mapping=(False,True)):
    self.path=path
    self.model=load_model(path)
    self.input=list(self.model.input_shape)
    self.input[0]=1
    self.mapping={
        0:mapping[0],
        1:mapping[1]
    }
  def preprocess(self,img):
    img = cv2.resize(img,tuple(self.input[1:3]))
    img = img.reshape(self.input)
    return img
  def predict(self,img):
    img=self.preprocess(img)
    pred=self.model.predict(img)
    i=np.argmax(pred)
    return (self.mapping[i],pred[0][i])
  
class binaryClassifier2:
  def __init__(self,path):
    self.path=path
    self.model=load_model(path)
    self.input=list(self.model.input_shape)
    self.input[0]=1
  def preprocess(self,img):
    img = cv2.resize(img,tuple(self.input[1:3]))
    img = np.stack((img,)*3, axis=-1)
    img = (img.reshape(self.input))/255.0
    return img
  def predict(self,img):
    img=self.preprocess(img)
    pred=self.model.predict(img)[0][0]
    return (pred>0.5,2*abs(0.5-pred))


class binaryClassifier3(binaryClassifier1):
  def __init__(self,path):
    super(binaryClassifier3, self).__init__(path)
  def preprocess(self,img):
    h,w=self.input[1:3]
    new=np.full((h,w),255,np.uint8)
    p=h//8
    new[p:-p,p:-p] = cv2.resize(img,(h-2*p,w-2*p))
    new = new.reshape(self.input)
    return new
  
