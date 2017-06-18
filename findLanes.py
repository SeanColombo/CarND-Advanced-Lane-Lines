import cv2
from IPython.display import HTML
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import numpy as np
import operator
import os
import pickle

"""
This script is for advanced lane-finding for Project 4 for Udacity's Self-Driving Car Engineering Nanodegree.

The initial flow will be based largely off of the Project 1 lanefinding video-processing, but with the
distortion-removal, perspective-transforms, and various gradients and thresholdings we learned in Lesson 14.

Potential future improvements for better and faster results:
    - Add sanity-check: make sure they're roughly parallel
    - Add santiy-check: make sure the curvatures are realistic based on the highway standards
    - Write the code to optimize lane-finding in sequential images (eg: use Line class). Should make it much faster.
    - Experiment with curvatures_via_convolutions() method. The spot to substitute it in, is in the code, commented out.
"""

# NOTE: These first 3 methods are not actively hit during the execution of the script as it stands, but they are
# a partial implementation that could be useful for any future attempts to 
def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(warped, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(warped.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
	    # convolve the window into the vertical slice of the image
	    image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
	    conv_signal = np.convolve(window, image_layer)
	    # Find the best left centroid by using past left center as a reference
	    # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
	    offset = window_width/2
	    l_min_index = int(max(l_center+offset-margin,0))
	    l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
	    l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
	    # Find the best right centroid by using past right center as a reference
	    r_min_index = int(max(r_center+offset-margin,0))
	    r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
	    r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
	    # Add what we found for that layer
	    window_centroids.append((l_center,r_center))

    return window_centroids
    
def curvatures_via_convolutions(binary_warped, do_output=False, image_name=""):
    # window settings
    window_width = 50 
    window_height = 100 # Break image into 9 vertical layers since image height is 720
    margin = 100 # How much to slide left and right for searching

    window_centroids = find_window_centroids(binary_warped, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(binary_warped)
        r_points = np.zeros_like(binary_warped)

        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width,window_height,binary_warped,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,binary_warped,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channel
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((binary_warped,binary_warped,binary_warped)),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
     
    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((binary_warped,binary_warped,binary_warped)),np.uint8)

    # Display the final results
    if do_output:
        plt.title('window fitting results')
        plt.savefig(os.path.join(OUT_DIR, "7_window-fitting-results"+image_name+".png"))
        plt.close()

    return binary_warped

    
    


def color_gradient_pipeline(img, do_output=False, image_name="", s_thresh=(100, 255), sx_thresh=(20, 100)):
    """
    Given a full-color image and optionally some threshold settings, will apply various thresholding
    based on color, Sobel derivatives, and the magnitude of gradients and combine them such that
    it will return a black-and-white binary image result where the regions that are likely to be
    lane-lines will be white and the image that is less likely to be lane-lines will be black.
    """
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    if do_output:
        cv2.imwrite(os.path.join(OUT_DIR, "3.10-hls-"+image_name+".png"), hls)

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    if do_output:
        cv2.imwrite(os.path.join(OUT_DIR, "3.20-sobel-"+image_name+".png"), scaled_sobel)
    
    # Threshold x gradient (magnitude of the gradient)
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel > sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    if do_output:
        write_binary_image(os.path.join(OUT_DIR, "3.30-thresh_x_gradient-"+image_name+".png"), sxbinary)
    
    # Threshold on S color channel
    if do_output:
        cv2.imwrite(os.path.join(OUT_DIR, "3.40-s_channel-"+image_name+".png"), s_channel)
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    if do_output:
        write_binary_image(os.path.join(OUT_DIR, "3.41-s_binary-"+image_name+".png"), s_binary)

    # Stack each channel
    # Just for seeing the different contributions in different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    if do_output:
        write_binary_image(os.path.join(OUT_DIR, "3.45-stacked_binaries-"+image_name+".png"), color_binary)
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    if do_output:
        write_binary_image(os.path.join(OUT_DIR, "3.50-color_binary-"+image_name+".png"), combined_binary)
    return combined_binary
    
def process_image(image, do_output=False, image_name=""):
    """
    Given an image (loaded from a file or a frame of a video), 
    process it to find the lane-lines and overlay them.

    image:      the full-color image (eg: from cv2.imread()).
    # WARNING: Using mtx & dist as globals, not params since fl_image doesn't allow us to pass parameters.
    #mtx:        cameraMatrics from cv2.calibrateCamera()
    #dist:       distortion-coefficients from cv2.calibrateCamera()
    do_output:  whether to output images of the various steps. Intended to be done doing
                for the static images but not for the videos (since there are a ton of frames).
    image_name: optional. Recommended when do_output is true. This will be used for debug
                and output-filenames related to this image.
    """

    image_name, image_extension = os.path.splitext(image_name)
    if do_output:
        # When doing debug-output, it's helpful to have a copy of the original image around for overlaying things
        # onto it.
        orig_image = np.copy(image)
        
    undist = cv2.undistort(image, mtx, dist, None, mtx)
    if do_output:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(undist)
        ax2.set_title('Undistorted Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(os.path.join(OUT_DIR, "2_undistorted_"+image_name+".png"))
        plt.close()

    # == Use color transforms, gradients, etc., to create a thresholded binary image. ==
    color_binary = color_gradient_pipeline(undist, do_output, image_name)
    if do_output:
        print("         Saving progress image for "+image_name+"...")
        write_binary_image(os.path.join(OUT_DIR, "3_color-gradient_"+image_name+".png"), color_binary)

    # == Perspective Transform ==
    # Apply a perspective transform to rectify binary image ("birds-eye view").
    # WE WILL FIND A TRAPEZOIDAL AREA OF INTEREST AND USE IT FOR THE PERSPECTIVE TRANSFORM!
    # We're defining a trapezoidal area at the center/bottom of the
    # screen (above the hood of the car though), in which to look.
    imgshape = color_binary.shape
    imgHeight = imgshape[0]
    imgWidth = imgshape[1]

    # === Configuration for perspective-transform source-trapezoid ===
    # trapezoidTopWidth is how wide the top of the trapezoid will be (got this by eyeballing it, and
    # tweaking until "straight lines" images transformed to be parallel). Initial attempts to just get the perspective-transformed
    # lines to be parallel, without first rendering the trapezoid lead to extremely short/wide trapezoid which was not optimal for
    # finding lane-lines. Once the trapezoid was rendered on the test-images, it became clear how it could be tweaked for better results.
    # This was definitely guess/test/revise and may not be appropriate for all cameras.
    # HIGHER VALUES MAKE THE TRANSFORMED LINES GET CLOSER AT THE TOP, LOWER trapezoidTopWidth VALUES MAKE THEM DIVERGE OUT AT THE TOP.
    trapazoidTopWidth = imgWidth * 0.10 # this is the width of the top line of the trapazoid
    trapazoidHeight = imgHeight * 0.32 # guess/test/revised to tune this number (this will be actual height of trapezoid)
    PADDING_FROM_SIDES = 50 # how many pixels from the side of camera before the left side of the trapazoid starts
    PADDING_FROM_BOTTOM = 40 # how many pixels to shave off the bottom of the image (basically, the hood of the car - just measured it in Gimp)

    # Trig to figure out the points in the trapezoid based on the configuration & image size:
    xOffset = (trapazoidTopWidth / 2) # distance that trapezoid top points will be from vertical center-line
    theta = math.atan( trapazoidHeight / (((imgWidth/2)-xOffset)-PADDING_FROM_SIDES) )
    topLeftX = ( (imgWidth/2) - xOffset )
    trapHeightCheck = ((topLeftX-PADDING_FROM_SIDES) * math.tan(theta))
    if (abs(trapHeightCheck - trapazoidHeight) > 0.001): # basically a unit-test for the trig I used ;)
        print("TRAPAZOID HEIGHT CHECK FAILED!")
        print("TRAP HEIGHT: ",trapazoidHeight)
        print("HEIGHT CHEK: ",trapHeightCheck)
    topLeftY = imgHeight - PADDING_FROM_BOTTOM - trapazoidHeight
    topRightX = ( (imgWidth/2) + xOffset )
    topRightY = topLeftY
    src = np.array([[
        (topLeftX, topLeftY), # top left
        (topRightX, topRightY), # top right
        (imgWidth-PADDING_FROM_SIDES, imgHeight-PADDING_FROM_BOTTOM), # bottom right
        (PADDING_FROM_SIDES, imgHeight-PADDING_FROM_BOTTOM) # bottom left
    ]], dtype=np.float32)
    if do_output:
        image_copy = np.copy(orig_image)
        cv2.polylines(image_copy,  np.int32([src]),True,(255,0,0), thickness=2)
        plt.imsave(os.path.join(OUT_DIR, "4.0-pre-warp-areaOfInterest-"+image_name+".png"), image_copy)
        plt.close()
    
    # Define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    # Basically fit the destination inside a square space in the image, inside of a certain amount of padding
    PADDING = 0 # it turns out that no padding was necessary. The lessons appeared to have horizontal padding but I couldn't think of any value that would provide.
    dst = np.float32([[PADDING, PADDING], # top left
                      [imgWidth-PADDING, PADDING], # top right
                      [imgWidth-PADDING, imgHeight-PADDING], # bottom right
                      [PADDING, imgHeight-PADDING]]) # bottom left
    # Use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Use cv2.warpPerspective() to warp the image to a top-down view
    binary_warped = cv2.warpPerspective(color_binary, M, (imgWidth, imgHeight), flags=cv2.INTER_LINEAR)
    if do_output:
        write_binary_image(os.path.join(OUT_DIR, "4.1-warped-"+image_name+".png"), binary_warped)
    image_height = binary_warped.shape[0]
    image_width = binary_warped.shape[1]
    
    # == FIND LINES ==
    # Detect lane pixels and fit to find the lane boundary...
    # === Histogram ===
    # STARTING POINT WAS DIRECTLY FROM LESSON 14-33. Modified from there.
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    if do_output:
        plt.plot(histogram)
        plt.savefig(os.path.join(OUT_DIR, "5_histogram_"+image_name+".png"))

    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # === Sliding window based on locations found in Histogram ===
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # === Visualizing the lane-finding ===
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    if do_output:
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, image_width)
        plt.ylim(image_height, 0)
        plt.savefig(os.path.join(OUT_DIR, "6_line_detection"+image_name+".png"))
        plt.close()

    # NOTE: IF IMPROVEMENTS WERE NEEDED, COULD EXPERIMENT WITH THIS ALTERNATIVE METHOD FROM LESSON 14-34
    #curvatures_via_convolutions(binary_warped, image_name)

    # == DETERMINE CURVATURE ==
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    if do_output:
        # Example values: 1926.74 1908.48
        print("\t\tCurve radii: ", left_curverad, right_curverad)

    # == CONVERT CURVATURE FROM PIXEL-SPACE TO WORLD-SPACE ==
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now the radius of curvature is in meters
    if do_output:
        # Example values: 632.1 m    626.2 m
        print("\t\tCurve radii in meters: ", left_curverad, 'm', right_curverad, 'm')

    # == WARP DETECTED LANE BOUNDARIES BACK ONTO THE ORIGINAL IMAGE ==
    # Warp the detected lane boundaries back onto the original image.
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    if do_output:
        plt.imsave(os.path.join(OUT_DIR, "8_lane_lines_"+image_name+".png"), result)
        plt.close()

    # == ADD REQUIRED ANNOTATIONS (CURVATURE & DIST FROM CENTER) ONTO THE IMAGE ==
    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    font = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 1.0
    FONT_COLOR = (255,0,0)
    # Find left-X/right-X of the bottom row of the image
    center_of_car = leftx[len(leftx)-1] + ((rightx[len(rightx)-1] - leftx[len(leftx)-1])/2)
    center_of_image = (image_width / 2)
    offset_from_center = (center_of_car - center_of_image) * xm_per_pix # find the offset-from-center in meters
    offset_from_center = round(offset_from_center, 2)
    # Need to putText right onto the image rather than using plt.text() because we need the annotations in the video-stream.
    cv2.putText(result, "Curvature radius: "+str(round(left_curverad, 2))+" meters", (10, 50), font, FONT_SCALE, FONT_COLOR, thickness=2)
    cv2.putText(result, "Car-center offset: "+str(offset_from_center)+" meters", (10,100), font, FONT_SCALE, FONT_COLOR, thickness=2)

    return result

def write_binary_image(file_name, img):
    """
    Since we use binary images (1's, 0's in a bitmap) but if we save them they look all-black (imwrite seems to
    expect them to be 0-255 still, this will output them as black and white.
    """
    output = img.copy()
    output[(img > 0)] = 255
    cv2.imwrite(file_name, output)


###############################################################
# SOME EXECUTING CODE FOR SETTING UP DIRECTORIES BEFORE ANY OF THE ACTUAL DATA GENERATION!
IN_DIR = "test_images"
CALIBRATION_DIR = "camera_cal"
OUT_DIR = "output_images"
VIDEO_IN_DIR = "."
VIDEO_OUT_DIR = "test_videos_output"

# Ensure the output directory for images/videos exist so that we can write to them.
if not os.path.exists(VIDEO_OUT_DIR):
    os.makedirs(VIDEO_OUT_DIR)
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
    
# == CAMERA CALIBRATION ==
print("Calibrating camera...")

# Camera calibration is time-consuming (10-12 seconds every time I try to execute the code) so
# the results are being saved to a pickle-file.
CALIBRATION_FILENAME = "calibration.p"

calibration_data = False
if os.path.isfile(CALIBRATION_FILENAME):
    calibration_data = pickle.load( open(CALIBRATION_FILENAME, "rb") )

if calibration_data:
    print("CALIBRATION DATA LOADED FROM PICKLE FILE! WILL NOT RE-CALIBRATE.")
    mtx = calibration_data["mtx"]
    dist = calibration_data["dist"]
else:
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
        print("Trying to calibrate with image ",str(image_number))
        img = cv2.imread(calibration_file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (NUM_X_CORNERS, NUM_Y_CORNERS), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

            # Draw and display the corners, save it to a file as a demonstration.
            img_with_corners = img.copy()
            cv2.drawChessboardCorners(img_with_corners, (NUM_X_CORNERS, NUM_Y_CORNERS), corners, ret)
            calibration_output_file_name = os.path.join(OUT_DIR, "z1_calibration_with_corners_"+str(image_number)+".png")
            cv2.imwrite(calibration_output_file_name, img_with_corners)

            # Calibrate the camera (each image should be making this calibration slightly better).
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            # == Demonstrate distortion-correction ==
            undist = cv2.undistort(img, mtx, dist, None, mtx)
            cv2.imwrite(os.path.join(OUT_DIR, "z2_undistorted_chess_board_"+str(image_number)+".png"), undist)
            print("Calibrated with image ",str(image_number))
            
    # Store the calibration to a pickle-file.
    calibration_data = {"mtx": mtx, "dist": dist}
    print("Storing calibration-data to file so that we won't have to re-calibrate on the next execution.")
    pickle.dump( calibration_data, open( CALIBRATION_FILENAME, "wb" ) )
print("Done calibrating camera.")


# Process and save each file that exists in the input directory.
print("Processing static images...")
files = os.listdir(IN_DIR)
for file_index in range(len(files)):
    fullFilePath = os.path.join(IN_DIR, files[file_index])
    
    # All of the image-processing is done in this call
    print("     Processing "+fullFilePath+"...")
    image = mpimg.imread(fullFilePath)
    image = process_image(image, do_output=True, image_name=files[file_index])
    
    # Take the processed image and save it to the output directory.
    saveFile = os.path.join(OUT_DIR, files[file_index])
    plt.imsave(saveFile, image)
print("Done processing static images.")


# Video processing
print("Processing video file...")
video_input_filename = os.path.join(VIDEO_IN_DIR, 'project_video.mp4')
video_output_filename = os.path.join(VIDEO_OUT_DIR, 'project_video.mp4')
clip1 = VideoFileClip(video_input_filename)
output_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
print("Writing video file...")
output_clip.write_videofile(video_output_filename, audio=False)

# print("Processing challenge video file...")
# video_input_filename = os.path.join(VIDEO_IN_DIR, 'challenge_video.mp4')
# video_output_filename = os.path.join(VIDEO_OUT_DIR, 'challenge_video.mp4')
# clip1 = VideoFileClip(video_input_filename)
# output_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# print("Writing video file...")
# output_clip.write_videofile(video_output_filename, audio=False)

# print("Processing harder-challenge video file...")
# video_input_filename = os.path.join(VIDEO_IN_DIR, 'harder_challenge_video.mp4')
# video_output_filename = os.path.join(VIDEO_OUT_DIR, 'harder_challenge_video.mp4')
# clip1 = VideoFileClip(video_input_filename)
# output_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# print("Writing video file...")
# output_clip.write_videofile(video_output_filename, audio=False)

print("Done!")
