'''
File: proj3.py
Author: John Gordon      Email: gordon8@umbc.edu
Section: CMSC 471        Meets: Mon/Wed @1
Description: This program takes in a filepath to a .jpg file and prints out
which object is in the picture. The program can determine if the .jpg is a 
smile, hat, dollar sign, hash, or a heart.

The README.txt file explains how to run this program (proj3.py)

Example data: 'Data' directory in this current directory

Google: support vector machine python

Things to do:
1. Convert bitmaps
2. Figure how to featurize
3. Convert to black and white (jpg could be in RGB when it's passed in)
4. Support vector machine implementation
'''
import sys
import os
import array
import numpy
from PIL import Image
#from sklearn.datasets import load_sample_image
from sklearn import svm


# Description: "Train" machine by loading all images from the Data folder into the 
# corressponding dataset.
def loadImages(clf, dataDir, data, target, picID):

    for jpg in os.listdir(dataDir):
        pic = []
        target = []
        data = []
        img = Image.open(dataDir+jpg).convert('L')
        imgArr = numpy.asarray(img)
        imgList = imgArr.tolist()

        for i in imgList:
            for j in i:
                pic.append(j)

        data.append(pic)
        target.append(picID)
    return data, target

def main(argv):
    
    ### Need to change path variable to directory of Data (directory of image data)
    path = '/home/jack/471/Projects/Proj3/Data'

    jpgFile = argv[1]    # passed in jpg for prediction image
    clf = svm.SVC(gamma=0.0001, C=100)

    # Target Array for fitting images in SVM. 0 = smiles, 1 = hat, 2 = hash, 3 = heart
    # 4 = dollar
    data = []
    target = []
    allData = [] # List of lists for all data of images
    allTargets = []
    # Convert jpg to pixel values, '1' converts to 0 or 1 for black or white respectively
    picture = Image.open(jpgFile).convert('L')    #'L' converts to pixel values from 0-255
    picArr = numpy.asarray(picture)
    picList = picArr.tolist()
    picL = []
    for i in picList:
        for j in i:
            picL.append(j)

    arrP = numpy.array(picL)
    # Loading all smiley faces from Data directory into a dataset
    smileDir = path + '/01/'
    data, target = loadImages(clf, smileDir, data, target, 0)

    n = 0
    for i in data:
        allData.append(i)
        allTargets.append(target[n])
        n += 1

    ### Loading hat pictures
    hatDir = path + '/02/'
    data, target = loadImages(clf, hatDir, data, target, 1)

    n = 0
    for i in data:
        allData.append(i)
        allTargets.append(target[n])
        n += 1

    ### Loading hash pictures
    hashDir = path + '/03/'
    data, target = loadImages(clf, hashDir, data, target, 2)

    n = 0
    for i in data:
        allData.append(i)
        allTargets.append(target[n])
        n += 1

    ### Loading heart pictures
    heartDir = path + '/04/'
    data, target = loadImages(clf, heartDir, data, target, 3)
    
    n = 0
    for i in data:
        allData.append(i)
        allTargets.append(target[n])
        n += 1

    ### Loading dollar pictures
    dollarDir = path + '/05/'
    data, target = loadImages(clf, dollarDir, data, target, 4)

    n = 0
    for i in data:
        allData.append(i)
        allTargets.append(target[n])
        n += 1


    # After all picture data has been loaded into lists/arrays
    arrD = numpy.array(allData)
    arrT = numpy.array(allTargets)

    # Let the SVM fit the picture data. Parameters: arrD - numpy array of all pictures.
    # arrT - The ID for the type of image, inline with arrD
    clf.fit(arrD, arrT)

    prediction = clf.predict(arrP)
    if(prediction[0] == 0):
        print("Smile")

    if(prediction[0] == 1):
        print("Hat")

    if(prediction[0] == 2):
        print("Hash")

    if(prediction[0] == 3):
        print("Heart")

    if(prediction[0] == 4):
        print("Dollar")

if __name__ == "__main__":
    main(sys.argv)
