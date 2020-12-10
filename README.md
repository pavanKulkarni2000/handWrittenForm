# Hand-Written Form Recognition
This module is used for hand-drawn form recognition.

### Table of Contents
You're sections headers will be used to reference location of destination.

- [Description](#description)
- [How To Use](#how-to-use)
- [References](#references)

---

## Description

In this module, we try to recognize various widgets of a form (certain identified classes of widgets).
<br><br>
First, we pass the scanned image of the form through the text detector. The detected text is erased from the binarized image of the form.<br>
To detect the widgets, we use a general method applicable to all the widgets, i.e. we use contour recognition method from OpenCV. Then after eliminating the redundant recognition, we use ML models to classify widgets. We have trained a binary classifier pertaining to each widget. Upon passing the image through all models, the prediction with highest probability is considered as the widget class.
<br><br>
### Sample dataset
<img src="https://github.com/pavanKulkarni2000/handWrittenForm/blob/main/images/dataset.png" alt="result"/>
<br><br><br>
An attempt to handwritten text recognition can also be seen here, where we build a custom handwritten OCR.<br>
A handritten character classifier is trained using the minst character dataset, with 62 classes(a-z,A-Z,0-9).<br>
The output of text detector is the image of a word. The word is segmented using contours finding methods from OpenCV. (<b>Note</b>: This segmentation works only if words are in the form of block letters i.e. characters are disjoint)<br>

### Model architecture
<img src="https://github.com/pavanKulkarni2000/handWrittenForm/blob/main/534a65a6-1d33-4b15-ac5c-b0b366f05f36.png" alt="result" width="400"/>

A different classifier can also be seen with only Capital characters classification(A-B).<br>
The mapping of the respective model classes can also be found.<br><br>

### Sample characters from EMINST dataset
<img src="https://github.com/pavanKulkarni2000/handWrittenForm/blob/main/images/chars.png" alt="result" width="400"/>

#### Sample form
<img src="https://github.com/pavanKulkarni2000/handWrittenForm/blob/main/images/input.jpg" alt="result" width="400"/>
<img src="https://github.com/pavanKulkarni2000/handWrittenForm/blob/main/images/result.jpg" alt="result" width="400"/>
<br><br>

#### Technologies

- OpenCV
- ML (CNN)

---

## How To Use

#### Installation
Clone the reposiorty, run the requirements.txt to install all the required python libraries.
<br>
Run the main.py with image path as argument to detect the widgets. The result is returned as a JSON.<br>

## References
- [Text Detector used is a reimplementation of the EAST text detector](https://github.com/SakuraRiven/EAST.git)
