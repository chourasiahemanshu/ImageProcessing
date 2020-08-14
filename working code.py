import numpy as np
import cv2
import random

np.seterr(divide='ignore', invalid='ignore')

sift = cv2.xfeatures2d.SIFT_create()

val1 = 'Rainier1.png'
val2 = 'Rainier2.png'
val3 = 'Rainier3.png'
val4 = 'Rainier4.png'
val5 = 'Rainier5.png'
val6 = 'Rainier6.png'

# Cimage1 = 'test1.JPG'
# Cimage2 = 'test2.JPG'
# Cimage3 = 'test3.JPG'

Cimage1 = 'image1.jpg'
Cimage2 = 'image2.jpg'
Cimage3 = 'image3.jpg'

# Cimage1 = 'img1.jpg'
# Cimage2 = 'img2.jpg'
# Cimage3 = 'img3.jpg'


boxes="Boxes.png"

# ---------Step1 : Calculating Harris Corner using in build function and then Computing sift descriptor ----------------------------------

def getHarrisKeyPointsAndDescriptor(img, gray):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)
    dst = cv2.cornerHarris(gray_img, 2, 3, 0.04)
    result_img = img.copy()
    # Threshold for an optimal value, it may vary depending on the image.
    result_img[dst > 0.01 * dst.max()] = [0, 0, 255]
    # for each dst larger than threshold, make a keypoint out of it
    keypoints = np.argwhere(dst > 0.01 * dst.max())
    keypoints = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints]
    imgR = cv2.drawKeypoints(img, keypoints, gray, color=(0, 0, 255))
    x,des = sift.compute(gray, keypoints)
    return keypoints, des ,imgR


# ----------- Matches the image using the keypoints and descriptors ----------------
def matchImages(kp1, kp2, des1, des2, img1, img2,stringName):
    bf = cv2.BFMatcher(crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # Draw first 10 matches.
    outputResult = cv2.drawMatches(img1, kp1, img2, kp2, matches[:300], img2)
    cv2.imwrite('2_'+ stringName + '.png', outputResult)
    # cv2.imshow("Matched Parts",outputResult)
    return matches



# Project Function as defined In the project Description
def project(Homography, x1, y1):
    mat = [x1, y1, 1]
    x, y, z = np.dot(Homography, mat)
    x2=x/z
    y2=y/z
    return int(x2) , int(y2)

#  Modified project which returns the float values of the x and y location
def projectPRaw(Homography, x1, y1):
    mat = [x1, y1, 1]
    x, y, z = np.dot(Homography, mat)
    x2=x/z
    y2=y/z
    return x2,y2

#  Modified project which returns the Array of float values of the source array
def projectAllPoints(Homography, sourcePoints):
    new_H_dst_points = []
    for points in sourcePoints:
        x, y = points
        mat = [x, y, 1]
        x, y, z = np.dot(Homography, mat)
        # print(x, y)
        new_H_dst_points.append((x / z, y / z))
    return new_H_dst_points
# -------------------------------------------------------

#-------------- Computes inlier counts ----------------------
def computeInlierCount(oldPoints, newPoints, threshold):
    inlierCount = []
    for i in range(0, len(oldPoints)):
        dist = ((oldPoints[i][0] - newPoints[i][0]) ** 2 + (oldPoints[i][1] - newPoints[i][1]) ** 2) ** 0.5
        if (dist < threshold):
            inlierCount.append((i))
    return inlierCount
# ------------------------------------------------------------

# Ransac takes input of the keypoints and matching points of the 2 images,
# I have broken this function  2 parts : ransacCalculations to perform all the Computations
# the other part takes the best inlier and computes new homography
def RANSAC(keyp1, keyp2, mPoints):
    ListOfInlier = []
    for i in range(500):
        temp = ransacCalculations(keyp1, keyp2, mPoints)
        ListOfInlier.append((len(temp), temp))
    ListOfInlier.sort(reverse=True)
    # print(ListOfInlier[0])
    newHomography = computeNewHomography(keyp1, keyp2, mPoints, ListOfInlier[0][1])
    final_matches_to_draw = calculateMathesOfInliers(ListOfInlier[0][1], mPoints)
    return final_matches_to_draw, newHomography



# ransacCalculations : takes random 4 points , finds homography and returns the list of inliers
def ransacCalculations(keypoints1, keypoints2, matchedPoints):
    #  Finding 4 random points to find Homography
    a = random.randint(0, len(matchedPoints) - 1)
    b = random.randint(0, len(matchedPoints) - 1)
    c = random.randint(0, len(matchedPoints) - 1)
    d = random.randint(0, len(matchedPoints) - 1)
    H = computeHomographyFor4Points(a, b, c, d, keypoints1, keypoints2, matchedPoints)
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matchedPoints])
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matchedPoints])
    # Projecting All the 4 Points
    new_H_dst_points = projectAllPoints(H, src_pts)
    # Finding The inliers
    inlierArray = computeInlierCount(dst_pts, new_H_dst_points, 0.7)
    return inlierArray


# -------------- Helper Functions of Ransac ----------------------------


def computeHomographyFor4Points(a, b, c, d, keypoints1, keypoints2, matchedPoints):
    src_pts_4 = []
    dst_pts_4 = []
    src_pts_4.append((keypoints1[matchedPoints[a].queryIdx]).pt)
    dst_pts_4.append((keypoints2[matchedPoints[a].trainIdx]).pt)
    src_pts_4.append((keypoints1[matchedPoints[b].queryIdx]).pt)
    dst_pts_4.append((keypoints2[matchedPoints[b].trainIdx]).pt)
    src_pts_4.append((keypoints1[matchedPoints[c].queryIdx]).pt)
    dst_pts_4.append((keypoints2[matchedPoints[c].trainIdx]).pt)
    src_pts_4.append((keypoints1[matchedPoints[d].queryIdx]).pt)
    dst_pts_4.append((keypoints2[matchedPoints[d].trainIdx]).pt)
    f_src = np.float32(src_pts_4)
    f_dst = np.float32(dst_pts_4)
    h, mask = cv2.findHomography(f_src, f_dst, 0)
    return h




def computeNewHomography(keypoints1, keypoints2, matchedPoints, listInliers):
    src_pts = []
    dst_pts = []
    # matches_final = []
    for i in listInliers:
        # matches_final.append(matchedPoints[i])
        src_pts.append((keypoints1[matchedPoints[i].queryIdx]).pt)
        dst_pts.append((keypoints2[matchedPoints[i].trainIdx]).pt)
    f_src = np.float32(src_pts)
    f_dst = np.float32(dst_pts)
    newH, mask = cv2.findHomography(f_src, f_dst, 0)
    return newH


def calculateMathesOfInliers(pointsToIterate, matches):
    selected_matches_to_return = []
    for i in pointsToIterate:
        selected_matches_to_return.append(matches[i])
    return selected_matches_to_return

# ----------------------------------------------------------------------------------------

# ----------------  Stitch Function uses the output of the RANSAC to stich to images --------

def stitch(image1, image2, homInv, hom):
    height = image1.shape[0]
    width = image1.shape[1]
    h2 = image2.shape[0]
    w2 = image2.shape[1]
    widthOffset=0
    heightOffset=0
    corner1Y,corner1X = project(homInv, 0, 0)
    corner2Y,corner2X = project(homInv, 0, h2)
    corner3Y,corner3X = project(homInv, w2, h2)
    corner4Y,corner4X = project(homInv, w2, 0)

    maxWidth  = max(corner1Y, corner2Y, corner3Y,corner4Y)
    maxHeight = max(corner1X, corner2X, corner3X,corner4X)
    if(maxWidth < width):
        maxWidth = width
    if(maxHeight < height):
        maxHeight = height

    widthOffset =min(corner1Y, corner2Y, corner3Y,corner4Y)
    heightOffset = min(corner1X, corner2X, corner3X,corner4X)
    if(widthOffset<0):
        widthOffset = abs(widthOffset)
    else:
        widthOffset = 0
    if(heightOffset<0):
        heightOffset = abs(heightOffset)
    else:
        heightOffset = 0
    # canvas = np.zeros(shape=(maxWidth, maxHeight))

    newHeight = abs(heightOffset)+maxHeight
    newWidth = abs(widthOffset)+maxWidth
    canvas = np.empty(( newHeight ,newWidth, 3), np.uint8)

    for i in range(0, newWidth ):
        for j in range(0, newHeight ):
            # print(j,i)

            try:
                canvas[j+abs(heightOffset)][i+abs(widthOffset)] = image1[j][i]
            except:
                pass

            x, y = projectPRaw(hom, i-abs(widthOffset), j-abs(heightOffset))
            try:
                if(x>0 and y>0 and y<h2 and x<w2):
                    a= cv2.getRectSubPix(image2,(1,1),(x,y))
                    canvas[j][i] = a
            except:
                pass
    return canvas

# -------------------------------------------------------------------------------------



def CalculateHarrisAndMatchesDoRansacAndStitchImages(img1, img2,stringName):
    gray1 = cv2.cvtColor(img1, 0)
    kp1, des1, image1 = getHarrisKeyPointsAndDescriptor(img1, gray1)
    cv2.imwrite('1b'+stringName+'.png', image1)
    # cv2.imshow("Matching Points", image1)
    gray2 = cv2.cvtColor(img2, 0)
    kp2, des2, image2 = getHarrisKeyPointsAndDescriptor(img2, gray2)

    cv2.imwrite('1b'+stringName+'.png', image2)
    # cv2.imshow("Matching Points", image2)

    matchedPoints = matchImages(kp1, kp2, des1, des2, img1, img2, stringName)

    final_matches_to_show, newHomography = RANSAC(kp1, kp2, matchedPoints)
    RansacResult = cv2.drawMatches(img1, kp1, img2, kp2, final_matches_to_show[:300], img2)
    cv2.imwrite('3_' + stringName + '.png', RansacResult)
    # cv2.imshow("Matched Parts", RansacResult)
    xh = np.linalg.inv(newHomography)
    c = stitch(img1, img2, xh, newHomography)
    cv2.imwrite('4_' + stringName + '.png', c)
    # cv2.imshow("Matched Parts", c)
    return c

def runRainierPhotosMainFunction():
    img1 = cv2.imread(val1)
    img2 = cv2.imread(val2)
    img3 = cv2.imread(val3)
    img4 = cv2.imread(val4)
    img5 = cv2.imread(val5)
    img6 = cv2.imread(val6)

    print("Stitching Image 1 and 2")
    canvas1 = CalculateHarrisAndMatchesDoRansacAndStitchImages(img1, img2 , "result1")
    print("Stitching Image 2 and 3")
    canvas2 = CalculateHarrisAndMatchesDoRansacAndStitchImages(canvas1, img3,"result2")
    print("Stitching Image 3 and 4")
    canvas3 = CalculateHarrisAndMatchesDoRansacAndStitchImages(canvas2, img4,"result3")
    print("Stitching Image 4 and 5")
    canvas4 = CalculateHarrisAndMatchesDoRansacAndStitchImages(canvas3, img5,"result4")
    print("Stitching Image 5 and 6")
    canvas5= CalculateHarrisAndMatchesDoRansacAndStitchImages(canvas4, img6,"result5")
    print("Printing Final Stitched Image")
    cv2.imshow("Canvas", canvas5)


def runCustomPhotosMainFunction():
    img1 = cv2.imread(Cimage1)
    img2 = cv2.imread(Cimage2)
    img3 = cv2.imread(Cimage3)

    print("Stitching Custom Image 1 and 2")
    canvas1 = CalculateHarrisAndMatchesDoRansacAndStitchImages(img1, img2,"custom_1")
    print("Stitching Custom Image 2 and 3")
    canvas2 = CalculateHarrisAndMatchesDoRansacAndStitchImages(canvas1, img3,"custom_2")
    print("Printing Final Stitched Image")
    cv2.imshow("Canvas", canvas2)


img3=cv2.imread(boxes)
gray3 = cv2.cvtColor(img3, 0)
keypoints3,descriptor,image3 = getHarrisKeyPointsAndDescriptor(img3,gray3)
cv2.imwrite('1a.png',image3)




# TODO Comment and Uncomment these 2 functions to see the results of the reiner and  custom made  photos
# runRainierPhotosMainFunction()
runCustomPhotosMainFunction()

cv2.waitKey(0)
cv2.destroyAllWindows()

