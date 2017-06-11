import cv2
from IPython.display import HTML
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import numpy as np
import operator
import os

"""
This script is for advanced lane-finding for Project 4 for Udacity's Self-Driving Car Engineering Nanodegree.

The initial flow will be based largely off of the Project 1 lanefinding video-processing, but with the
distortion-removal, perspective-transforms, and various gradients and thresholdings we learned in Lesson 14.
"""

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, lane_min_y, lane_max_y, origImg, color=[255, 0, 0], thickness=5):
    """
    Separates line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, averages the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    The results are opaque, transparency is applied (with weighted_img) to the
    results of the hough_lines function.
    """
    
    # If we set too small of a target area, or otherwise poorly tuned our parameters, it's
    # possible that there were no lines found.
    if lines is None:
        print("There were no lines detected in this frame.")
        return
    
    # Identify lines (by slope) as being left or right lane-lines, and find
    # an average of the high and low points to create an average line for each
    # lane.
    numLeftLines = 0
    lowLeftTotals = [0, 0]
    highLeftTotals = [0, 0]
    numRightLines = 0
    lowRightTotals = [0, 0]
    highRightTotals = [0, 0]
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = ((y2-y1)/(x2-x1))
            if(slope < 0): # since y value is at bottom of image, negative slope is left lane lines
                #print("Left slope: ",slope)
                numLeftLines += 1
                if(x1 < x2):
                    lowLeftTotals = list(map(operator.add, lowLeftTotals, [x1, y1]))
                    highLeftTotals = list(map(operator.add, highLeftTotals, [x2, y2]))
                else:
                    lowLeftTotals = list(map(operator.add, lowLeftTotals, [x2, y2]))
                    highLeftTotals = list(map(operator.add, highLeftTotals, [x1, y1]))
            elif(slope > 0):
                #print("Right slope: ",slope)
                numRightLines += 1
                if(x1 < x2):
                    highRightTotals = list(map(operator.add, highRightTotals, [x1, y1]))
                    lowRightTotals = list(map(operator.add, lowRightTotals, [x2, y2]))
                else:
                    highRightTotals = list(map(operator.add, highRightTotals, [x2, y2]))
                    lowRightTotals = list(map(operator.add, lowRightTotals, [x1, y1]))
            else:
                # Horizontal lines (slope = 0) will be ignored.
                continue
                    
    # NOTE: Sometimes at this point, we have found no lines on one or more sides. If that's the case, we just don't
    # draw that line. One other valid approach could be to remember the previous right line and use
    # that... until we've gone X iterations in a row without new information (then throw some exception).
    # This would give us a reasonable approximation of the missing lines on frames that don't have one. For
    # now we just skip drawing the line that doesn't have points.
    if((numLeftLines ==0)  or (numRightLines == 0)):
        print("One side's lane-line is missing from this frame.  LEFT LINES: ",numLeftLines," RIGHT LINES: ",numRightLines)
        #saveFile = os.path.join(OUT_DIR, "missing_lines_"+str(numLeftLines)+"_"+str(numRightLines)+".jpg") # will keep overwriting the same image
        #plt.imsave(saveFile, origImg)
        #plt.figure()
        #plt.imshow(origImg) # show every image in the notebook for debugging


    # Average all lane-lines to find a representative line for each lane.
    if(numLeftLines > 0):
        lowLeftPoint = list(map(operator.truediv, lowLeftTotals, [numLeftLines,numLeftLines]))
        highLeftPoint = list(map(operator.truediv, highLeftTotals, [numLeftLines,numLeftLines]))
    if(numRightLines > 0):
        lowRightPoint = list(map(operator.truediv, lowRightTotals, [numRightLines,numRightLines]))
        highRightPoint = list(map(operator.truediv, highRightTotals, [numRightLines, numRightLines]))

    # Extrapolate each lane-line to fill the full height of the lane.
    # We will do this by finding the intersection between each lane-line and the lane_min_y (to find top point of line)
    # and the intersection b/w the lane-line and lane_max_y (bottom point of the line).
    if(numLeftLines > 0):
        [lowLeftPoint, highLeftPoint] = extrapolateLineToMinMaxY(lowLeftPoint, highLeftPoint, lane_min_y, lane_max_y)
    if(numRightLines > 0):
        [lowRightPoint, highRightPoint] = extrapolateLineToMinMaxY(lowRightPoint, highRightPoint, lane_min_y, lane_max_y)
    
    # cv2.line requires ints and rounding is more accurate than just casting
    if(numLeftLines > 0):
        lowLeftPoint = [ round(elem) for elem in lowLeftPoint ]
        lowLeftPoint = list(map(int, lowLeftPoint))
        highLeftPoint = [ round(elem) for elem in highLeftPoint ]
        highLeftPoint = list(map(int, highLeftPoint))
    if(numRightLines > 0):
        lowRightPoint = [ round(elem) for elem in lowRightPoint ]
        lowRightPoint = list(map(int, lowRightPoint))
        highRightPoint = [ round(elem) for elem in highRightPoint ]
        highRightPoint = list(map(int, highRightPoint))

    # Draw each averaged-out, extrapolated lane-line. Using the number of lines as the thickness
    if(numLeftLines > 0):
        cv2.line(img, (lowLeftPoint[0], lowLeftPoint[1]), (highLeftPoint[0], highLeftPoint[1]), color, thickness)
    if(numRightLines > 0):
        cv2.line(img, (lowRightPoint[0], lowRightPoint[1]), (highRightPoint[0], highRightPoint[1]), color, thickness)
    
def extrapolateLineToMinMaxY(p1, p2, min_y, max_y):
    """
    Given points p1 and p2, will return new points of a line that go between min_y and max_y. This
    will be used so that our lane-lines are extrapolated to the entire mask-area that we were looking at.
    
    The return-value will be a list containing two lists: one for each point of the new line.
    """
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    slope = ((y2-y1)/(x2-x1))
    yInterceptOfLine = (y1 - (slope*x1))
    
    # Based on the equation for intersection of a horizontal line and another line (that I derived on paper):
    min_x = ((min_y - yInterceptOfLine) / slope)
    max_x = ((max_y - yInterceptOfLine) / slope)

    return [ [min_x, min_y], [max_x, max_y] ]
    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, min_y, max_y):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.  This has been modified from its initial state
    to extrapolate those lines to extend between min_y and max_y (eg: to fill the height of
    the lane).
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines, min_y, max_y, img)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, a=0.8, ß=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * a + img * ß + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, ß, λ)


# Given an image (eg: loaded from a file or a video-frame)
# processes it to overlay lane-lines on top of the image
# as semi-transparent red lines. Returns the modified image.
def process_image_OLD(image):
    gray = grayscale(image)
    
    # Apply Gaussian smoothing & Canny edge detection
    blur_gray = gaussian_blur(gray, kernel_size=3)

    # Define our parameters for Canny and apply
    edges = canny(blur_gray, low_threshold=50, high_threshold=150)
    
    # == MASK OUT THE TRAPAZOIDAL AREA OF INTEREST ==
    # We're defining a trapazoidal area at the center/bottom of the
    # screen, in which to look. triangleHeight and trapazoidTopWidth are
    # the parameters to manually tune, to control the size of the trapazoid.
    imgshape = image.shape
    imgHeight = imgshape[0]
    imgWidth = imgshape[1]
    triangleHeight = imgHeight * 0.44 # guess/test/revised to tune this number
    trapazoidTopWidth = imgWidth * 0.1 # how wide the top of the trapezoid will be
    xOffset = (trapazoidTopWidth / 2) # dist that trapazoid top points will be from vertical center-line
    # Trig to figure out the points in the trapazoid based on the configuration & image size:
    theta = math.atan( triangleHeight / (imgWidth/2) )
    topLeftX = ( (imgWidth/2) - xOffset )
    topLeftY = imgHeight - (topLeftX * math.tan(theta))
    topRightX = ( (imgWidth/2) + xOffset )
    topRightY = topLeftY

    # We have the points... use the trapezoid as a mask.
    vertices = np.array([[
        (0, imgHeight),
        (topLeftX, topLeftY),
        (topRightX, topRightY),
        (imgWidth, imgHeight)
    ]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # == HOUGH TRANSFORM ==
    # Define the Hough transform parameters - going with recommended
    # weightings used in Lession 1:15.
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 30     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 100 #minimum number of pixels making up a line
    max_line_gap = 160    # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap, topLeftY, imgHeight)
    
    # Overlay the hough lines on the original image.
    combined_image = weighted_img(line_image, image)

    # The original image will now be returned with semitransparent lane-lines overlaid on it.
    return combined_image
    
def process_image(image):
    """
    Given an image (loaded from a file or a frame of a video), 
    process it to find the lane-lines and overlay them.
    
    NOTE: This function expects full-color images (eg: the return value of cv2.imread().
    """

    # TODO: IMPLEMENT HERE!!
    
    # REMAINING PROJECT STEPS:
    # Use color transforms, gradients, etc., to create a thresholded binary image.
    # Apply a perspective transform to rectify binary image ("birds-eye view").
    # Detect lane pixels and fit to find the lane boundary.
    # Determine the curvature of the lane and vehicle position with respect to center.
    # Warp the detected lane boundaries back onto the original image.
    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


    
    return image




###############################################################
# SOME EXECUTING CODE FOR SETTING UP DIRECTORIES BEFORE ANY OF THE ACTUAL DATA GENERATION!
IN_DIR = "test_images"
CALIBRATION_DIR = "camera_cal"
OUT_DIR = "output_images"
VIDEO_IN_DIR = "test_videos"
VIDEO_OUT_DIR = "test_videos_output"

# Ensure the output directory for images/videos exist so that we can write to them.
if not os.path.exists(VIDEO_OUT_DIR):
    os.makedirs(VIDEO_OUT_DIR)
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

    
# == CAMERA CALIBRATION ==
print("Calibrating camera...")
NUM_X_CORNERS = 9 # number of interior corners horizontally
NUM_Y_CORNERS = 6 # number of interior corners vertically

# The object points and image points will be built-up accross all
# of our calibration images, so as we add more images, the calibration
# should get slightly better for a while.
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ...,(7,5,0)
objp = np.zeros((NUM_X_CORNERS*NUM_Y_CORNERS, 3), np.float32)
objp[:,:2] = np.mgrid[0:NUM_X_CORNERS,0:NUM_Y_CORNERS].T.reshape(-1,2) # x, y coordinates

# Iterate through each of the chessboard calibration images.
for image_number in range(1,21):
    calibration_file_name = os.path.join(CALIBRATION_DIR, "calibration"+str(image_number)+".jpg")
    img = cv2.imread(calibration_file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (NUM_X_CORNERS, NUM_Y_CORNERS), None)
    if ret == True:
        imgpoints.append(corners)
        objpoints.append(objp)

        # Draw and display the corners, save it to a file as a demonstration.
        img_with_corners = img.copy()
        cv2.drawChessboardCorners(img_with_corners, (NUM_X_CORNERS, NUM_Y_CORNERS), corners, ret)
        calibration_output_file_name = os.path.join(OUT_DIR, "1_calibration_with_corners_"+str(image_number)+".png")
        cv2.imwrite(calibration_output_file_name, img_with_corners)

        # Calibrate the camera (each image should be making this calibration slightly better).
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # == Demonstrate distortion-correction ==
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite(os.path.join(OUT_DIR, "2_undistorted_chess_board_"+str(image_number)+".png"), img)

print("Done calibrating camera.")

# Process and save each file that exists in the input directory.
print("Processing static images...")
files = os.listdir(IN_DIR)
for fileIndex in range(len(files)):
    fullFilePath = os.path.join(IN_DIR, files[fileIndex])
    
    # All of the image-processing is done in this call
    image = mpimg.imread(fullFilePath)
    image = process_image(image)
    
    # Take the processed image and save it to the output directory.
    saveFile = os.path.join(OUT_DIR, files[fileIndex])
    plt.imsave(saveFile, image)
    
    # The files are already saved... also show the image in the notebook.
    plt.figure()
    #plt.imshow(image)
print("Done processing static images.")


# Video processing
# print("Processing video file...")
# video_input_filename = os.path.join(VIDEO_IN_DIR, 'project_video.mp4')
# video_output_filename = os.path.join(VIDEO_OUT_DIR, 'project_video.mp4')
# clip1 = VideoFileClip(video_input_filename)
# output_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# print("Writing video file...")
# output_clip.write_videofile(video_output_filename, audio=False)
# print("Done!")
