from PIL import Image
import numpy as np
import cv2
import widgets
import utils
import textUtils
import classes
import textRecognize
import textDetect
import widgetModels
from statistics import mean

def init_models():
  #initializing models
  models={
    "textDetector":textDetect.TextDetector(),
    "textrec":textRecognize.TextRecognizer(),
    "widgetClassifier":widgetModels.WidgetClassifier()
  }
  return models

def image_to_json(imgc,models):

  #remove noise
  imgc = cv2.fastNlMeansDenoisingColored(imgc)

  #get gray image
  gray=cv2.cvtColor(imgc,6)
  #remove noise
  gray = cv2.fastNlMeansDenoising(gray,None,10,7,21)
  h,w=gray.shape

  #get binary image
  img= cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
              cv2.THRESH_BINARY,11,2)


  #initializing array for all the components of the form
  fields=[]


  #detect text
  tboxes=models["textDetector"].boxes(Image.fromarray(imgc))
  #recognize text
  for tbox in tboxes:
    x,y,w,h=map(int,tbox)
    new=img[y:y+h,x:x+w]
    res=models["textrec"].recognize(new)
    if res is not None:
      fields.append(classes.Label(res[0],tbox,float(res[1])))
  #calculate average text height, used as reference
  avg_t_h=mean([b[3] for b in tboxes])
  fields=textUtils.groupLabels(fields,avg_t_h) 


  #erasing text
  utils.erase(tboxes,img)


  #detect all components remaining on the form
  contours, hierarchy = cv2.findContours(255-img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  boxes=sorted([cv2.boundingRect(c) for c in contours], key =lambda x:x[2],reverse=True )

  #cleaning
  if (0,0,w,h) in boxes:
    boxes.remove((0,0,w,h))
  temp=[]
  for b in boxes:
    if b[2]<=8 or b[2]*b[3]<150 :
      temp.append(b)
  boxes=[b for b in boxes if b not in temp]
  for b in boxes:
    l=[x for x in boxes if utils.is_inside(x,b,5)]
    boxes=[x for x in boxes if x not in l]

  #recognize components
  lines=[]
  for i in range(len(boxes)):
    x,y,w,h=boxes[i]
    res=models["widgetClassifier"].classify(img[y:y+h,x:x+w],avg_t_h)
    if res is None:
      continue
    
    Ftype=res[1] 

    if Ftype in ["textBox","dropDown","date"]:
      ls=[f for f in fields if f.fieldType == "Label" and utils.is_inside(f.getBbox(),boxes[i],avg_t_h/3)]
      if ls:
        fields=[f for f in fields if f not in ls]
        fields.append(classes.Button(boxes[i],textUtils.mergeLabels(ls)))
      else:
        if Ftype == "textBox":
          fields.append(classes.TextField(boxes[i]))
        elif Ftype == "dropDown":
          fields.append(classes.DropDown(boxes[i]))
        else:
          fields.append(classes.Date(boxes[i]))
    
    elif Ftype == "line":
      ls=[f for f in fields if f.fieldType == "Label" and widgets.is_underline(f.getBbox(),boxes[i],avg_t_h//2,avg_t_h//2)]
      if len(ls) != 0:
        fields=[f for f in fields if f not in ls]
        ls=textUtils.mergeLabels(ls).toTitle()
        ls.headerLine=classes.Line(boxes[i])
        fields.append(ls)
      else:
        lines.append(classes.Line(boxes[i]))

    elif Ftype == "checkBox":
      fields.append(classes.CheckBox(boxes[i]))
    
    elif Ftype == "radio":
      fields.append(classes.Radio(boxes[i]))
    
    elif Ftype == "table":
      fields.append(classes.Table(boxes[i]))
      
  # processing the widgets

  #collecting all the lines
  for l in lines:
    # To recognize lines as fill-in-the blank text-box
    # comparing the line with adjacent text,
    # which is made one fourth it's height so as to match the line height
    textFieldLine=False
    for t in tboxes :
      bbox1=(t[0]+(t[2]*3)//4,t[1],t[2]//4,t[3]);
      bbox2=l.getBbox();
      if utils.is_left(bbox1,bbox2) or utils.is_right(bbox1,bbox2):
        textFieldLine=True
        break
    if textFieldLine:
      lines.remove(l)
      ls=list(l.getBbox())
      ls[3]=avg_t_h
      fields.append(TextField(ls))
      

  lines.sort(key=lambda x:x.BoundingBox['Top'])
  # grouping lines into multiline textArea
  groups=[]
  while lines:
    ls=[lines.pop(0)]
    while True:
      temp=[line for line in lines if utils.is_above(ls[-1].getBbox(),line.getBbox(),avg_t_h)]
      if temp:
        ls.append(min(temp,key=lambda x:x.BoundingBox['Top'] - ls[-1].BoundingBox['Top']))
        lines.remove(ls[-1])
      else:
        break
    groups.append(ls)

  #form text area out of lines
  for group in groups:
    if len(group)==1:
      fields.append(group[0])
    else:
      fields.append(classes.TextArea(utils.getBbox(group),len(group)))



  for f in [f for f in fields if f.fieldType=="Table"]:
    x,y,w,h=f.getBbox()
    f.rows=len(widgets.getRows(img[y:y+h,x:x+w]))
    f.cols=len(widgets.getColums(img[y:y+h,x:x+w]))
    if f.rows == 0 or f.cols == 0:
      fields.remove(f)
  for f in [f for f in fields if f.fieldType=="Date"]:
    x,y,w,h=f.getBbox()
    f.blocks=len(widgets.getColums(img[y:y+h,x:x+w]))
    if f.blocks == 0:
      fields.remove(f)
    #collecting the row and column count in the table

  #rectifying the heights and sorting
  fields=utils.makeHeights(fields,avg_t_h/2)
  #association of related components
  components=utils.getComponents(fields,avg_t_h)
  #breaking of the form into sections
  sections=utils.getSections(components)

  new=classes.form(img.shape)
  new.sections=sections
  cv2.imwrite("result.jpg",new.printForm())
  return new.getJSON()

