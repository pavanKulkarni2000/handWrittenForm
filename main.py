import argparse
import os
import form
import cv2
import sys

# Create the parser
parser = argparse.ArgumentParser(description='form creation')

# Add the arguments
parser.add_argument('Path',
                       type=str,
                       help='the path to image')

input_path = parser.parse_args().Path

if not os.path.isfile(input_path):
    print('The path specified does not exist')
    sys.exit()

try:
  imgc = cv2.imread(input_path)
except:
  print("cannot open image")
  sys.exit()

models=form.init_models()

JsonResult=form.image_to_json(imgc,models)
print(JsonResult)