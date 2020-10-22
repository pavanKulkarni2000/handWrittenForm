import cv2
import numpy as np
import utils

def getRows(img):
  h,w=img.shape
  bw=np.full((h+10,w+10),255,dtype=np.uint8)
  bw[5:-5,5:-5]=img
  
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,4))
  abh = cv2.erode(bw, kernel, iterations = 1)

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,1))
  abh = cv2.dilate(abh, kernel, iterations = 3)


  contoursh, hierarchy = cv2.findContours(255-abh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  # for c in contoursh:
  #   x,y,w,h=cv2.boundingRect(c)
  #   print(str((x,y,w,h)))

  boxes=[cv2.boundingRect(c) for c in contoursh]
  for b in boxes:
    if b[2]<3*b[3]:
      boxes.remove(b)
    if b in boxes:
      ls= [boxes.pop(boxes.index(bx)) for bx in boxes if utils.is_inLine(b,bx,max(min(b[3],bx[3]),3)//2) ]
      boxes.append(utils.getBbox(ls))
  boxes.sort(key= lambda x:x[1])
  ls=[]
  for i in range(1,len(boxes)):
    ls.append((boxes[i][1]+boxes[i][3]//2)-(boxes[i-1][1]+boxes[i-1][3]//2))
  return ls
    

    


def getColums(img):
  h,w=img.shape
  bw=np.full((h+10,w+10),255,dtype=np.uint8)
  bw[5:-5,5:-5]=img

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,1))
  abv = cv2.erode(bw, kernel, iterations = 1)

  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
  abv = cv2.dilate(abv, kernel, iterations = 3)

  contoursv, hierarchy = cv2.findContours(255-abv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  # for c in contoursv:
  #   x,y,w,h=cv2.boundingRect(c)
  #   print(str((x,y,w,h)))

  boxes=[cv2.boundingRect(c) for c in contoursv]
  for b in boxes:
    if b[3]<3*b[2]:
      boxes.remove(b)
    if b in boxes:
      ls= [boxes.pop(boxes.index(bx)) for bx in boxes if utils.is_withinWidth(b,bx,max(min(b[2],bx[2]),3)//2) ]
      boxes.append(utils.getBbox(ls))
  boxes.sort()
  ls=[]
  for i in range(1,len(boxes)):
    ls.append((boxes[i][0]+boxes[i][2]//2)-(boxes[i-1][0]+boxes[i-1][2]//2))
  return ls


def is_underline(label,line,v=3,h=3):
  x1,y1,w1,h1=label
  x2,y2,w2,h2=line
  if (x1>x2 or abs(x2-x1)<h) and ((x1+w1)<(x2+w2) or abs((x1+w1)-(x2+w2))<h) and abs(y2-(y1+h1))<v:
    return True
  return False
