import json
import numpy as np
import cv2
from collections import OrderedDict
from statistics import mean


def is_inside(bbox1,bbox2):
  x1,y1,w1,h1=bbox1
  x2,y2,w2,h2=bbox2
  if (x1<x2 and y1<y2 and x1+w1>x2+w2 and y1+h1>y2+h2) or\
   (x1>x2 and y1>y2 and x1+w1<x2+w2 and y1+h1<y2+h2):
    return True
  return False


class form:
  def __init__(self,dim=(-1,-1)):
    self.Geometry={
            "Height": dim[0],
            "Width": dim[1]
    }
    self.sections=[]
    self.Json=None


  def printForm(self):

    form_image = np.full((self.Geometry["Height"],self.Geometry["Width"],3), 255, dtype=np.uint8)
    cv2.rectangle(form_image,(10,10,self.Geometry["Width"]-10,self.Geometry["Height"]-10),(0,0,0),1)
    fields=[]
    for s in self.sections:
      for c in s.components:
        fields+=c.fields
    for f in fields:
      #Labels and Titles
      if f.fieldType in ["Label"] :
        i=(f.BoundingBox["Height"])/40
        form_image = cv2.putText(form_image,f.text, (f.BoundingBox["Left"]+5,f.BoundingBox["Top"]+f.BoundingBox["Height"]-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX ,\
                                           fontScale=i, color=(0,0,0), thickness=int(i),lineType= cv2.LINE_AA,bottomLeftOrigin=False) 
      elif f.fieldType in ["Title"] :
        i=(f.BoundingBox["Height"])/40
        form_image = cv2.putText(form_image,f.text, (f.BoundingBox["Left"]+5,f.BoundingBox["Top"]+f.BoundingBox["Height"]-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX ,\
                                           fontScale=i, color=(0,0,0), thickness=int(i)+1,lineType= cv2.LINE_AA,bottomLeftOrigin=False) 

        form_image=cv2.line(form_image,(f.BoundingBox["Left"]+5,f.BoundingBox["Top"]+f.BoundingBox["Height"]+10),\
        (f.BoundingBox["Left"]+f.BoundingBox["Width"],f.BoundingBox["Top"]+f.BoundingBox["Height"]+10),\
        (0,0,0),2)
      elif f.fieldType in ["TextField", "CheckBox","DropDown"]:
        x,y,w,h=f.getBbox()
        form_image =cv2.rectangle(form_image,(x,y,w,h),(0,0,0),2)
        if f.fieldType == "DropDown":
          cv2.line(form_image,(x+w-h,y),(x+w-h,y+h),(0,0,0),2)
          cv2.rectangle(form_image,(x+w-h,y,h,h),(220,220,220),-1)
          pts=np.array([(x+w-h+h//4,y+h//4),(x+w-h+h//2,y+h*3//4),(x+w-h+h*3//4,y+h//4)])
          cv2.drawContours(form_image,[pts], 0, (0,0,0), -1)
      elif f.fieldType in ["Radio"]:
        form_image=cv2.circle(form_image,(f.Circle["center"]["x"],f.Circle["center"]["y"]),f.Circle["radius"],(0,0,0),2)
      elif f.fieldType in ["Date"]:
        x,y,w,h=f.getBbox()
        if f.blocks<=2:
          cv2.rectangle(form_image,(x,y,w//2-2,h),(0,0,0),2 )
          cv2.line(form_image,(x+w//4,y),(x+w//4,y+h),(0,0,0),2)
          cv2.rectangle(form_image,(x+w//2,y,w//2-2,h),(0,0,0),2 )
          cv2.line(form_image,(x+w*3//4,y),(x+w*3//4,y+h),(0,0,0),2)
        else:
          cv2.rectangle(form_image,(x,y,w//4-2,h),(0,0,0),2 )
          cv2.line(form_image,(x+w//8,y),(x+w//8,y+h),(0,0,0),2)
          cv2.rectangle(form_image,(x+w//4,y,w//4-2,h),(0,0,0),2 )
          cv2.line(form_image,(x+w*3//8,y),(x+w*3//8,y+h),(0,0,0),2)
          cv2.rectangle(form_image,(x+w//2,y,w//2-2,h),(0,0,0),2 )
          cv2.line(form_image,(x+w*5//8,y),(x+w*5//8,y+h),(0,0,0),2)
          cv2.line(form_image,(x+w*6//8,y),(x+w*6//8,y+h),(0,0,0),2)
          cv2.line(form_image,(x+w*7//8,y),(x+w*7//8,y+h),(0,0,0),2)
      
      elif f.fieldType in   ["Table"]:
        x,y,w,h=f.getBbox()
        cv2.rectangle(form_image,(x,y,w,h),(0,0,0),2 )
        for i in range(1,f.rows):
          cv2.line(form_image,(x,y+(h//f.rows)*i),(x+w,y+(h//f.rows)*i),(0,0,0),2)
        for i in range(1,f.cols):
          cv2.line(form_image,(x+(w//f.cols)*i,y),(x+(w//f.cols)*i,y+h),(0,0,0),2)
      elif f.fieldType in   ["Button"]:
        x,y,w,h=f.getBbox()
        cv2.rectangle(form_image,(x,y,w,h),(0,0,0),2 )
        cv2.rectangle(form_image,(x,y,w,h),(220,220,220),-1)
        f=f.label
        i=(f.BoundingBox["Height"])/40
        form_image = cv2.putText(form_image,f.text, (f.BoundingBox["Left"]+5,f.BoundingBox["Top"]+f.BoundingBox["Height"]-5), fontFace=cv2.FONT_HERSHEY_SIMPLEX ,\
                                           fontScale=i, color=(0,0,0), thickness=int(i),lineType= cv2.LINE_AA,bottomLeftOrigin=False) 
    return form_image

  def getJSON(self):

    if self.Json is None:
      d=self.__dict__.copy()
      d.pop('Json')
      d['sections']=[f.getJSON() for f in self.sections]
      self.Json=d
      
    return json.dumps(self.Json,indent=4)

    
class GenericField:
  def __init__(self, bbox=(0, 0, 0, 0),fieldType=None):
    self.fieldType=fieldType
    self.BoundingBox = {
            "Left": int(bbox[0]),  
            "Top": int(bbox[1]), 
            "Width": int(bbox[2]),
            "Height": int(bbox[3])
            }
  def getBbox(self):
    return tuple(self.BoundingBox.values())
  def setBbox(self,bbox):
    self.BoundingBox = {
            "Left": int(bbox[0]),  
            "Top": int(bbox[1]), 
            "Width": int(bbox[2]),
            "Height": int(bbox[3])
            }
  def getJSON(self):
    return dict(self.__dict__)
  def __str__(self):
    return str(self.__dict__)
  def __repr__(self):
    return self.__str__()

class Component(GenericField):
  def __init__(self, bbox,fields):
    super().__init__(bbox,"Component")
    self.fields=sorted(fields,key=lambda x:(x.BoundingBox['Top'],x.BoundingBox['Left']))
  def getJSON(self):
    d= dict(self.__dict__).copy()
    d['fields']=[f.getJSON() for f in self.fields]
    return d

class Section(GenericField):
  def __init__(self, bbox,components):
    super().__init__(bbox,"Section")
    self.components=sorted(components,key=lambda x:(x.BoundingBox['Top'],x.BoundingBox['Left']))
  def getJSON(self):
    d= dict(self.__dict__).copy()
    d['components']=[c.getJSON() for c in self.components]
    return d
  

class Label(GenericField):
  def __init__(self, text=None, bbox=(0, 0, 0, 0), conf=0):
    super(Label, self).__init__(bbox,"Label")
    self.text=text
    self.confidence=float(round(conf,2))
  def toTitle(self):
    return Title(self.text,self.getBbox(),self.confidence)

class Title(GenericField):
  def __init__(self, text=None , bbox=(0, 0, 0, 0), conf=0, line=None):
    super(Title, self).__init__(bbox,"Title")
    self.text=text
    self.confidence=float(round(conf,2))
    self.headerLine=line
  def getJSON(self):
    d= dict(self.__dict__).copy()
    d['headerLine']=self.headerLine.getJSON()
    return d

  

class Button(GenericField):
  def __init__(self, bbox=(0, 0, 0, 0), label=None):
    super(Button, self).__init__(bbox,"Button")
    self.label=label
  def getJSON(self):
    d= dict(self.__dict__).copy()
    d['label']=self.label.getJSON()
    return d
  
class Radio(GenericField):
  def __init__(self,bbox=None,circle=None):
    if bbox is not None:
      super(Radio, self).__init__(bbox,"Radio")
      self.Circle = {"center": {"x": bbox[0] + bbox[2]//2, "y": bbox[1] + bbox[3]//2}, "radius": min(bbox[2],bbox[3])//2}
    else:
      bbox=(circle[0]-circle[2]-1,circle[1]-circle[2]-1,2*circle[2]+2,2*circle[2]+2)
      super(Radio, self).__init__(bbox,"Radio")
      self.Circle = {"center": {"x": int(circle[0]), "y": int(circle[1])}, "radius": int(circle[2])}
   

class TextField(GenericField):
  def __init__(self, bbox=( 0, 0, 0, 0)):
    super(TextField, self).__init__(bbox,"TextField")

class TextArea(GenericField):
  def __init__(self, bbox=( 0, 0, 0, 0),lines=None):
    super(TextArea, self).__init__(bbox,"TextArea")
    self.lines=lines


class Line(GenericField):
  def __init__(self, bbox=( 0, 0, 0, 0)):
    super(Line, self).__init__(bbox,"Line")

class DropDown(GenericField):
  def __init__(self, bbox=( 0, 0, 0, 0)):
    super(DropDown, self).__init__(bbox,"DropDown")

class CheckBox(GenericField):
  def __init__(self,bbox=(0, 0, 0, 0)):
    super(CheckBox, self).__init__(bbox,"CheckBox")

class Date(GenericField):
  def __init__(self, bbox=(0, 0, 0, 0),blocks=0):
    super(Date, self).__init__(bbox,"Date")
    self.blocks=blocks

class Table(GenericField):
  def __init__(self,bbox=(0, 0, 0, 0),rows=0,cols=0):
    super(Table, self).__init__(bbox,"Table")
    self.rows=rows
    self.cols=cols
