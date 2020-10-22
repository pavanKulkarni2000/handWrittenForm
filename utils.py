import numpy as np
import statistics
from classes import *
import cv2


#returns background color of the bounding box
def getBackground(pic,b,p=3):
  x,y,w,h=b
  arr=[]
  if x-p>=0:
    arr=(pic[y:y+w,x-p]).tolist()
  if y-p>=0:
    arr=arr+(pic[y-p,x:x+w]).tolist()
  if x+w+p<pic.shape[1]:
    arr=arr+(pic[y:y+w,x+w+p]).tolist()
  if y+h+p<pic.shape[0]:
    arr=arr+(pic[y+h+p,x:x+w]).tolist()

  while True:
    #to handle multiple mode error
    try:
      return statistics.mode(arr)
    except:
      if arr:
        arr.pop(0)
      else:
        return 0


#
#       erase a componet from the binary image 
#
# arguments: list of fields to be erased 
#            binary image
#            aditional padding, default 1
#            make the compnent white(255) or black(0)
#            use background to erase 
#
def erase(fields,pic,p=1,val=255,background=False):
  #list of fields(which extends the genericfields type) given:
  try:
    for f in fields:
      x,y,w,h=f.getBbox()
      if background:
        val=getBackground(pic,(x,y,w,h))
      pic[y-p:y+h+p, x-p:x+w+p]=val

  #list of boundingboxes given
  except:
    for box in fields:
      x,y,w,h=map(int,box)
      if background:
        val=getBackground(pic,(x,y,w,h))
      pic[y-p:y+h+p, x-p:x+w+p]=val
  return



#
#       removes bounding box only
#
def remove(fields,OTSU):
  #list of fields(which extends the genericfields type) given:
  try:
    for f in fields:
      OTSU=cv2.rectangle(OTSU,f.getBbox(),(255,255,255),2)

  #list of boundingboxes given
  except:
    for b in fields:
      OTSU=cv2.rectangle(OTSU,b,(255,255,255),2)
  return



#
#     returns the truth value of is one bounding box inside the other
#
def is_inside(bbox1,bbox2,b=5):
  x1,y1,w1,h1=bbox1
  x2,y2,w2,h2=bbox2
  if (x1+b)>x2 and (y1+b)>y2 and (x1+w1)<(x2+w2) and (y1+h1)<(y2+h2) :
    return True
  if x1>x2 and y1>y2 and (x1+w1)<(x2+w2+b) and (y1+h1)<(y2+h2+b) :
    return True
  return False

#horizontal_overlap
def overlap(bbox1,bbox2):
  x1,y1,w1,h1=bbox1
  x2,y2,w2,h2=bbox2
  x_left = max(x1,x2)
  x_right = min(x1+w1,x2+w2)
  if x_right < x_left:
    return False
  else:
    return True



def is_inLine(bbox1,bbox2,d=3):
  x1,y1,w1,h1=bbox1
  x2,y2,w2,h2=bbox2
  if abs(y1-y2)<d and abs(y1+h1-(y2+h2))<d:
    return True
  return False


# is bbox1 to right of bbox2
def is_right(bbox1,bbox2,d=3):
  if is_inLine(bbox1,bbox2) and abs(bbox2[0]+bbox2[2]-bbox1[0])<d:
    return True
  return False

# is bbox1 to left of bbox2
def is_left(bbox1,bbox2,d=3):
  if is_inLine(bbox1,bbox2,d) and abs(bbox1[0]+bbox1[2]-bbox2[0])<d:
    return True
  return False

# is bbox1 above of bbox2
def is_above(bbox1,bbox2,d=3):
  if is_withinWidth(bbox1,bbox2,d//2) and abs(bbox1[1]+bbox1[3]-bbox2[1])<d :
    return True
  return False

# is bbox1 below of bbox2
def is_below(bbox1,bbox2,d=3):
  if is_withinWidth(bbox1,bbox2,d//2) and abs(bbox2[1]+bbox2[3]-bbox1[1])<d :
    return True
  return False


def is_neighbour(bbox1,bbox2,d):
  x1,y1,w1,h1=bbox1
  x2,y2,w2,h2=bbox2
  if (((x1<x2 or abs(x1-x2)<d ) and x1+w1>x2+w2) or ((x2<x1 or abs(x2-x1)<d ) and x2+w2>x1+w1)) and (abs(y1-y2-h2)<d or abs(y2-y1-h1)<d) :
    return True
  if abs(y1-y2)<d/2  and (abs(x1-x2-w2)<d or abs(x2-x1-w1)<d) :
    return True
  return False

def is_withinWidth(bbox1,bbox2,d=3):
  x1,y1,w1,h1=bbox1
  x2,y2,w2,h2=bbox2
  if abs(x1-x2)<d and abs(x1+w1-(x2+w2))<d:
    return True
  return False

def makeHeights(fields,d=8):
  fields=sorted(fields,key=lambda x: x.BoundingBox['Top'])
  for i in range(1,len(fields)):
    if abs(fields[i].BoundingBox['Top'] -fields[i-1].BoundingBox['Top'])<d:
      fields[i].BoundingBox['Top']=fields[i-1].BoundingBox['Top']
  return sorted(fields,key=lambda x: (x.BoundingBox['Top'],x.BoundingBox['Left']))

def getSections(components):

  sectionHeaders=[c.fields[0] for c in components if c.fields[0].fieldType=='Title']
  sectionHeaders.sort(key=lambda x:x.BoundingBox['Top'])

  sections=[]
  while sectionHeaders:
    header=sectionHeaders.pop(-1)
    section=[c for c in components if c.BoundingBox['Top'] >= header.BoundingBox['Top']]
    [components.remove(c) for  c in section]

    sections.insert(0,Section(getBbox(section),section))
  if components:
    sections.insert(0,Section(getBbox(components),components))
  return sections




def getComponents(fields,ref):
  tfields=sorted([f for f in fields if f.fieldType == "Label"],key=lambda x: (x.BoundingBox['Top'],x.BoundingBox['Left']))
  fields=sorted([f for f in fields if f not in tfields  ], key=lambda x: (x.BoundingBox['Top'],x.BoundingBox['Left']))
  final=[]
  for w in fields:
    if w.fieldType in ['Line','Title']:
      final.append({w})
      continue
    ls=set([f for f in tfields if is_neighbour(w.getBbox(),f.getBbox(),ref)])
    if len(ls)>1:

      ls_=set([f for f in ls if f.BoundingBox['Left']<w.BoundingBox['Left']])
      if len(ls_) >0 :
        ls_.add(w)
        final.append(ls_)
        continue
      
      ls_=set([f for f in ls if f.BoundingBox['Top']<w.BoundingBox['Top']])
      if len(ls_) >0 :
        ls_.add(w)
        final.append(ls_)
        continue
      
      ls_=set([f for f in ls if f.BoundingBox['Left']>w.BoundingBox['Left']])
      if len(ls_) >0 :
        ls_.add(w)
        final.append(ls_)
        continue
      
      ls.add(w)
      final.append(ls)
      continue
    else:
      ls.add(w)
      final.append(ls)
      continue
  for f in final:
    if f in final:
      ls=[e for e in final if not e.isdisjoint(f)]
      if len(ls) >= 1:
        for e in ls:
          final.remove(e)
          f.update(e)
        final.append(f)
  final=[Component(getBbox(ls),list(ls)) for ls in final]
  return final


def getBbox(fields):
  #list of fields(which extends the genericfields type) given:
  try:
    l=[f.getBbox() for f in fields]
    i0=min(l,key= lambda i:i[0])
    i1=min(l,key= lambda i:i[1])
    i2=max(l,key= lambda i:i[0]+i[2])
    i3=max(l,key= lambda i:i[1]+i[3])
    return (i0[0],i1[1],i2[0]+i2[2]-i0[0],i3[1]+i3[3]-i1[1])

  #list of boundingboxes given
  except:
    l=fields
    i0=min(l,key= lambda i:i[0])
    i1=min(l,key= lambda i:i[1])
    i2=max(l,key= lambda i:i[0]+i[2])
    i3=max(l,key= lambda i:i[1]+i[3])
    return (i0[0],i1[1],i2[0]+i2[2]-i0[0],i3[1]+i3[3]-i1[1])