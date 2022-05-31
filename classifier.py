import cv2
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

x = np.load("image (1).npz")['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A','B','C','D','E', 'F','G','H','I','J','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nClasses = len(classes)  

xtrain, xtest, ytrain, ytest = train_test_split(x,y, random_state=9, train_size=7500, test_size=2500)
xtrain_scaled = xtrain/255.0
xtest_scaled = xtest/255.0
clf = LogisticRegression(solver = "saga", multi_class="multinomial").fit(xtrain_scaled,ytrain)
ypred = clf.predict(xtest_scaled) 
accuracy = accuracy_score(ytest, ypred)
print(accuracy)

def getPrediction(image):
    impill = Image.open(image)
    imagebw = impill.convert("L")
    imagebw_resized = imagebw.resize((28,28),Image.ANTIALIAS)
    pixelFilter = 20
    minPixel = np.percentile(imagebw_resized, pixelFilter)
    imagebw_resized_inverted_scaled = np.clip(imagebw_resized - minPixel, 0,255)
    maxPixel = np.max(imagebw_resized)
    imagebw_resized_inverted_scaled = np.asarray(imagebw_resized_inverted_scaled)/maxPixel
    
    testSample = np.array(imagebw_resized_inverted_scaled).reshape(1,784)
    testPred = clf.predict(testSample)
    return(testPred[0])
