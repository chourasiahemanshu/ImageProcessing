import numpy as np
import cv2 as cv
np.seterr(divide='ignore', invalid='ignore')


val = 'Boxes.png'
val1 = 'Boxes.png'


# ----------------------------------------  Part1 : Calculating Harris Corner  ----------------------------------
def harrisCornerPoint(imgstring,thresholdPercentage):
    img = cv.imread(imgstring, 0)
    imgCopy = cv.imread(imgstring)
    height = img.shape[0]
    width = img.shape[1]
    margin = 2

    RMatrix = np.zeros(shape=(height, width))
    # ******************* Applying Sobel Operator to take the Gradient **********************
    dx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=5)
    dy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=5)

    ixx = dx**2
    iyy = dy**2
    ixy = dx*dy
    # ******************* Applying Gaussian Filter to Smooth the Image **********************
    Ixx = cv.GaussianBlur(ixx,(5,5),0)
    Iyy = cv.GaussianBlur(iyy,(5,5),0)
    Ixy = cv.GaussianBlur(ixy,(5,5),0)

    # ******************* Calculating Harris Matrix (Method 1) **********************
    det =  (Ixx * Iyy)-(Ixy**2)
    trace = (Ixx + Iyy)
    RMatrix = det/trace

            # ******************* Calculating Harris Matrix (Method 2 : Both Works, but getting better result with Method 1) **********************
            # for y in range(2, height):
            #     for x in range(2, width):
            #         windowIxx = Ixx[y - margin:y + margin + 1, x - margin:x + margin + 1]
            #         windowIxy = Ixy[y - margin:y + margin + 1, x - margin:x + margin + 1]
            #         windowIyy = Iyy[y - margin:y + margin + 1, x - margin:x + margin + 1]
            #
            #         Sxx = windowIxx.sum()
            #         Syy = windowIyy.sum()
            #         Sxy = windowIxy.sum()
            #
            #         det =(Sxx * Syy)-(Sxy**2)
            #         trace =  Sxx+Syy
            #         R = det/(trace)
            #         RMatrix[y][x]=R

    # ******************* Calculating Threshold and Making it Max value in the Neighbourhood **********************
    rmax = np.max(RMatrix)
    threshold1 = rmax * thresholdPercentage
    print("Threshold for 1 :" , threshold1)
    keypoint=[]
    # RMatrix=[]
    for y in range(2, height-2,5):
        for x in range(2, width-2,5):
            window = RMatrix[y - margin:y + margin + 1, x - margin:x + margin + 1]
            val = np.amax(window)
            window.fill(0)
            RMatrix[y][x] = val

            if RMatrix[y][x]> threshold1:
                keypoint.append(cv.KeyPoint(x,y,1,-1,0,0,-1))

    # -------------------------------- part 2 : Feature Descriptor -----------------------------------

    rad = (np.arctan2(dy, dx))
    angles = np.degrees(rad) % 360
    magnitude = ((Ixx) + (Iyy)) ** 0.5
    superMainBin = []
    skipping = 0

    # The first loop is to iterate over all my keypoints
    for y in range(8, height - 8):
        for x in range(8, width - 8):
            if RMatrix[y][x] > threshold1:
                # creating 16x16 window
                try:
                    mainBin = []
                    for a in range(y - 8, y + 8, 4):
                        for b in range(x - 8, x + 8, 4):
                            smallBin = {}
                            for a in range(0, 8):
                                smallBin[a] = 0
                            # iterating over 4x4 window
                            for m in range(a - 2, a + 2):
                                for n in range(b - 2, b + 2):
                                    remainder = angles[n][m] % 45
                                    if (remainder == 0):
                                        binval = int(angles[n][m] / 45)
                                        smallBin[binval] = magnitude[n][m]
                                    else:
                                        binval = int(angles[n][m] / 45)
                                        smallBin[binval] = ((45 - remainder) / 45) * magnitude[n][m]
                                        smallBin[(binval + 1) % 8] = ((remainder) / 45) * magnitude[n][m]
                            mainBin.append(list(smallBin.values()))
                    normalisedMainBin = mainBin / np.linalg.norm(mainBin)
                    trimmedAndNormalizedMainBin = np.clip(normalisedMainBin, 0, 0.2)

                    superMainBin.append(trimmedAndNormalizedMainBin)
                except:
                    skipping += 1
    return imgCopy,keypoint,superMainBin





def featureMatatching(descriptor1 ,  descriptor2):
    matchedDescriptors = []
    RatioDistance = []
    for i,descVal in enumerate(descriptor1):
        ssd=[]
        allSSD = []
        for j,otherVal in enumerate(descriptor2):
            dist = np.sum(np.subtract(descVal,otherVal)**2)
            allSSD.append(dist)
            if(dist<0.85):
                 ssd.append(dist)
        if(len(ssd)!=0):
            minVal = min(ssd)
            secondLowestVal = np.partition(ssd, 2)[1]
            ratio = minVal/secondLowestVal
            RatioDistance.append(ratio)
            matchedDescriptors.append(cv.DMatch(i, allSSD.index(minVal), minVal))
    return matchedDescriptors , RatioDistance


# Calculating Harris Corner and Descriptors for 2 Values and its Threshold Value
image_1, key_points_1, descriptor1 = harrisCornerPoint(val, 0.8)
image_2, key_points_2, descriptor2  = harrisCornerPoint(val1, 0.7)

# Calculating Matched Descriptors and the Ratio Distance between them
matchedDescriptorsList , RatioDist = featureMatatching(descriptor1, descriptor2)

im1_keypoints = cv.drawKeypoints(image_1, key_points_1, image_1, color=(0, 0, 255))
im2_keypoints = cv.drawKeypoints(image_2, key_points_2, image_2, color=(0, 0, 255))

cv.imshow('Image 1', im2_keypoints)
cv.moveWindow('Image 1', 200,200)
cv.imshow('Image 2', im1_keypoints)
cv.moveWindow('Image 2', 750,200)

img3 = cv.drawMatches(image_1, key_points_1, image_2, key_points_2, matchedDescriptorsList[:15], None, flags=2)
cv.imshow("Matching Points",img3)
cv.waitKey(0)
cv.destroyAllWindows()
