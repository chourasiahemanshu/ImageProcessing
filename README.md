# ImageProcessing
OpenCV project to create Panorama Image by combining multiple images.


1. Computed Harris Corner Using build in functions

2. Calculated Sift descriptor using in built function

3.Bellow are the major functions used to perform the RANSAC and compute the inliers with function definition 

-------------------------------------------------------------------------------------------
Major Functions As per the Project Requirement:

getHarrisKeyPointsAndDescriptor(img, gray)
Calculating Harris Corner using in build function and then Computing sift descriptor 

matchImages(kp1, kp2, des1, des2, img1, img2)
Matches the image using the keypoints and descriptors

project(Homography, x1, y1)
Project Function as defined In the project Description: it has 3 varients : 
one of them projects and gives output in integers
the other function return the float and negative value
the other function returns the array of new points

projectPRaw(Homography, x1, y1)
Modified project which returns the float values of the x and y location

projectAllPoints(Homography, sourcePoints)
Modified project which returns the Array of float values of the source array


Computes inlier counts
computeInlierCount(oldPoints, newPoints, threshold)
I have taken the threshold as 0.7


Ransac takes input of the keypoints and matching points of the 2 images,It performs the iterations 500 times to get good result
I have broken this function  2 parts : ransacCalculations to perform all the Computations
the other part takes the best inlier and computes new homography
RANSAC(keyp1, keyp2, mPoints)



ransacCalculations(keypoints1, keypoints2, matchedPoints)
ransacCalculations : takes random 4 points , finds homography and returns the list of inliers

Stitch Function uses the output of the RANSAC to stich to images
stitch(image1, image2, homInv, hom)
I have not used any blending technique.
It normaly copy pastes the pixel from source to the destination

---------------------------------------------------------------------------------------

Things to be noted :
(The code should give best image in first try but do rerun it if it doesnt do that... )
Run the working code. It will run the 2 functions  (you need to toggle the comments in these functions)
runCustomPhotosMainFunction()
runRainierPhotosMainFunction()

These functions execute all the tasks.
All the photos should be in the same file to run this
The output images are computes in the same folder.


Programming Assignment #2
----------------------------------------------------------------------------------------------------------
DESCRIPTION
In this assignment, you will write code to detect discriminating features in an image and find the best matching
features in other images. Because features should be reasonably invariant to translation, rotation (plus
illumination and scale if you do the extra credit), you'll use a feature descriptor discussed during lecture and you'll
evaluate its performance on a suite of benchmark images. As part of the extra credit you'll have the option of
creating your own feature descriptors.
In the Project, you will apply your features to automatically stitch images into a panorama.


Feature matching
Now that you've detected and described your features, the next step is to write code to match them, i.e., given a
feature in one image, find the best matching feature in one or more other images. This part of the feature
detection and matching component is mainly designed to help you test out your feature descriptor. The simplest
approach is the following: write a procedure that compares two features and outputs a distance between them.
For example, you could simply sum the absolute value of differences between the descriptor elements. You could
then use this distance to compute the best match between a feature in one image and the set of features in
another image by finding the one with the smallest distance.
Two distance measures you should implement are:
1. A threshold on the match score. This is called the SSD distance.
2. (score of the best feature match)/(score of the second best feature match). This is called
the "ratio test".

Testing
Using the OpenCV API you can load in a set of images, view the detected features using cv::drawKeypoints(...),
and visualize the feature matches that your algorithm computes using cv::drawMatches(...).
We are providing a set of benchmark images to be used to test the performance of your algorithm as a function of
different types of controlled variation (i.e., rotation, scale, illumination, perspective, blurring).




