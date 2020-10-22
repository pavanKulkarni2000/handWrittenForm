from classes import Label, Title
import utils
import statistics



def mergeLabels(ls):
  ls.sort(key = lambda x:x.BoundingBox['Left'])
  label=ls[0]
  label.confidence=statistics.mean([l.confidence for l in ls])
  for i in range(1, len(ls)):
    l=ls[i]
    p=ls[i-1]
    if l.BoundingBox['Left'] - p.BoundingBox['Left'] - p.BoundingBox['Width'] < l.BoundingBox['Height']//4:
      label.text+=l.text
      label.text=processWord(label.text)
    else:
      label.text+=" "+l.text
  label.setBbox(utils.getBbox(ls))
  return label

def groupLabels(fields,ref):
  fields=utils.makeHeights(fields)
  grouped=[]
  levels=set([f.BoundingBox['Top'] for f in fields])
  for level in levels:
    ls=[f for f in fields if f.BoundingBox['Top']==level]
    ls.sort(key = lambda x:x.BoundingBox['Left'])
    i=0;n=len(ls);
    while i<n:
      group=[ls[i]]
      i+=1
      while i<n and ls[i].BoundingBox['Left']-ls[i-1].BoundingBox['Left']-ls[i-1].BoundingBox['Width']<2*ref:
        group.append(ls[i])
        i+=1
      grouped.append(mergeLabels(group))
  return grouped
    

def processWord(word):
  if word.isalnum():
    word=word.replace('0','O')
    word=word.replace('5','S')
  
  caps=len([a for a in word if a.isupper()])
  small=len(word)-caps

  if small>=2:
    if word[0].isupper():
      word=word.capitalize()
    else:
      word=word.lower()
  else:
    word=word.upper()

  return word
