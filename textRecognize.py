import cv2
import numpy as np
from tensorflow.keras.models import load_model
import utils
from statistics import mean
import textUtils

W = 700
H = 250

def getCharacters(img):
  kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
  img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel,iterations=3)
  h,w=img.shape
  img=cv2.resize(img,(1000,1000))
  contours, hierarchy = cv2.findContours(255-img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  boxes=[list(cv2.boundingRect(c)) for c in contours]
  if (0,0,1000,1000) in boxes:
    boxes.remove((0,0,1000,1000))
  for b in boxes:
    l=[x for x in boxes if utils.is_inside(x,b,5)]
    boxes=[x for x in boxes if x not in l]
    b[0]=int(b[0]*w/1000)
    b[1]=int(b[1]*h/1000)
    b[2]=int(b[2]*w/1000)
    b[3]=int(b[3]*h/1000)
  return boxes

class TextRecognizer:
  def __init__(self,path='./text',capsOnly=False):
    if capsOnly:
      self.charClassifier=load_model(path+'/CAPS-char.model')
      mapFile=open(path+"/CAPS-mapping.txt")
    else:
      self.charClassifier=load_model(path+'/char.model')
      mapFile=open(path+"/mapping.txt")
    self.map={}
    for line in mapFile.readlines():
      i,c=line.split()
      self.map.update({int(i):c})

  def getChar(self,char):
    ret2,char = cv2.threshold(char,0,255,cv2.THRESH_OTSU)
    # kernel=np.ones((5,5),np.uint8)
    # char = cv2.erode(char,kernel,iterations = p+4)
    if char is None:
      return None
    h,w=char.shape
    d=abs(w-h)
    if d>1:
      if w>h:
        new=np.full((w,w), 255, dtype=np.uint8)
        new[d//2:d//2+h,0:w]=char
      else:
        r=w/h
        new=np.full((h,h), 255, dtype=np.uint8)
        new[0:h,d//2:d//2+w]=char
    else:
      new=char
    res=[]
    for p in range(6):
      char=cv2.resize(new,(28-2*p,28-2*p))
      new=np.full((28,28), 255, dtype=np.uint8)
      new[p:28-p,p:28-p]=char
      char=new
      char=char.reshape(1,28,28,1)
      pred=self.charClassifier.predict(char)
      i=np.argmax(pred)
      ch=self.map[i]
      res.append((ch,pred[0][i]))
    res=max(res,key=lambda x:x[1])
    return res if res[1]>0.55 else None
        
    while res:
      ch=max(res.items(),key=lambda x:x[1][0]*x[1][1])
      if ch[1][0]<0.6:
        res.pop(ch[0])
      else:
        break
    return (ch[0],ch[1][0]) if ch[1][0]>0.6 else None
    
  def parseCharacters(self,img):
    h,w=img.shape
    scale=1000//max(h,w)
    offset=scale//10
    new=np.full((2*offset+h*scale,2*offset+w*scale),255,dtype=np.uint8)
    new[offset:offset+h*scale,offset:offset+w*scale]=cv2.resize(img,(0,0),fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    img=new
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img=cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel,iterations=2)
    contours, hierarchy = cv2.findContours(255-img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes=[list(cv2.boundingRect(c)) for c in contours]
    boxes.sort(key=lambda x:(x[2],x[3]),reverse=True)
    for b in boxes:
      if b in boxes:
        if b[3]/scale < h//10:
          boxes.remove(b)
          continue
        l=[x for x in boxes if utils.is_inside(x,b,5)]
        boxes=[x for x in boxes if x not in l]
        l=[x for x in boxes if utils.is_withinWidth(x,b,min(b[2],x[2])//2)]
        boxes=[x for x in boxes if x not in l]
        b=list(utils.getBbox(l))
        b[0]=(b[0]-offset)//scale
        b[1]=(b[1]-offset)//scale
        b[2]=b[2]//scale
        b[3]=b[3]//scale
        boxes.append(b)
    return boxes

  #recognizes a word
  def recognize(self,img):
    _,img = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    characters=self.parseCharacters(img)
    characters.sort()
    preds=[]
    for x,y,w,h in characters:
      res=self.getChar(img[y:y+h,x:x+w])
      if res is not None:
        preds.append(res)
    if len(preds)==0:
      return None
    
    word=''.join([a for a,_ in preds])
    word=textUtils.processWord(word)
    conf=mean([a for _,a in preds])
    
    return (word,conf)
    


