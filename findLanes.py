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

# Python 3 has support for cool math symbols. I don't advise using these :P this function was
# provided in an Udacity lesson though.
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
    
    
def color_gradient_pipeline(img, s_thresh=(100, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    cv2.imwrite(os.path.join(OUT_DIR, "3.10-hls-.png"), hls) # TODO: REMOVE?

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    cv2.imwrite(os.path.join(OUT_DIR, "3.20-sobel-.png"), scaled_sobel) # TODO: REMOVE?
    
    # Threshold x gradient (magnitude of the gradient)
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel > sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    write_binary_image(os.path.join(OUT_DIR, "3.30-thresh_x_gradient-.png"), sxbinary) # TODO: REMOVE?
    
    # Threshold on S color channel
    cv2.imwrite(os.path.join(OUT_DIR, "3.40-s_channel-.png"), s_channel) # TODO: REMOVE?
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    write_binary_image(os.path.join(OUT_DIR, "3.41-s_binary-.png"), s_binary) # TODO: REMOVE?

    # Stack each channel
    # Just for seeing the different contributions in different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    write_binary_image(os.path.join(OUT_DIR, "3.45-stacked_binaries-.png"), color_binary) # TODO: REMOVE?
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    write_binary_image(os.path.join(OUT_DIR, "3.50-color_binary-.png"), combined_binary) # TODO: REMOVE?
    return combined_binary
    
def process_image(image, mtx, dist, do_output=False, image_name=""):
    """
    Given an image (loaded from a file or a frame of a video), 
    process it to find the lane-lines and overlay them.
    
    image:      the full-color image (eg: from cv2.imread()).
    mtx:        cameraMatrics from cv2.calibrateCamera()
    dist:       distortion-coefficients from cv2.calibrateCamera()
    do_output:  whether to output images of the various steps. Intended to be done doing
                for the static images but not for the videos (since there are a ton of frames).
    image_name: optional. Recommended when do_output is true. This will be used for debug
                and output-filenames related to this image.
    """
    
    image_name, image_extension = os.path.splitext(image_name)
    
    # == Use color transforms, gradients, etc., to create a thresholded binary image. ==
    color_binary = color_gradient_pipeline(image)
    if do_output == True:
        print("         Saving progress image for "+image_name+"...")
        write_binary_image(os.path.join(OUT_DIR, "3_color-gradient_"+image_name+".png"), color_binary)

    # == Perspective Transform ==
    # Apply a perspective transform to rectify binary image ("birds-eye view").
    # WE WILL FIND A TRAPAZOIDAL AREA OF INTEREST AND USE IT FOR THE PERSPECTIVE TRANSFORM!
    # We're defining a trapazoidal area at the center/bottom of the
    # screen, in which to look. triangleHeight and trapazoidTopWidth are
    # the parameters to manually tune, to control the size of the trapazoid.
    imgshape = color_binary.shape
    imgHeight = imgshape[0]
    imgWidth = imgshape[1]
    PADDING_FROM_BOTTOM = 40 # how many pixels to shave off the bottom of the image (basically, the hood of the car - just measured it in Gimp)
    triangleHeight = imgHeight * 0.44 # guess/test/revised to tune this number
    # How wide the top of the trapezoid will be (got this by tweaking until "straight lines" images transformed to be parallel.
    # This was definitely guess/test/revise and may not be appropriate for all cameras.
    # TODO: CONTINUE TO TWEAK THIS VALUE!!! 0.35 IS VERY CLOSE, BUT OVERLAY A GRID ON IT TO SEE IF IT SHOULD BE TWEAKED MORE!!!
    # HIGHER VALUES MAKE THEM GO OUT AT THE TOP, LOWER MAKE THEM COME BACK IN.
    trapazoidTopWidth = imgWidth * 0.35
    xOffset = (trapazoidTopWidth / 2) # dist that trapezoid top points will be from vertical center-line
    # Trig to figure out the points in the trapezoid based on the configuration & image size:
    theta = math.atan( triangleHeight / (imgWidth/2) )
    topLeftX = ( (imgWidth/2) - xOffset )
    topLeftY = imgHeight - (topLeftX * math.tan(theta))
    topRightX = ( (imgWidth/2) + xOffset )
    topRightY = topLeftY
    src = np.array([[
        (topLeftX, topLeftY), # top left
        (topRightX, topRightY), # top right
        (imgWidth, imgHeight-PADDING_FROM_BOTTOM), # bottom right
        (0, imgHeight-PADDING_FROM_BOTTOM) # bottom left
    ]], dtype=np.float32)
    # Define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    # Basically fit the destination inside a square space in the image, inside of a certain amount of padding
    PADDING = 0 # TODO: EXPERIMENT GETTING THIS TO 0
    dst = np.float32([[PADDING, PADDING], # top left
                      [imgWidth-PADDING, PADDING], # top right
                      [imgWidth-PADDING, imgHeight-PADDING], # bottom right
                      [PADDING, imgHeight-PADDING]]) # bottom left
    # Use cv2.getPerspectiveTransform() to get M, the transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Use cv2.warpPerspective() to warp your image to a top-down view
    binary_warped = cv2.warpPerspective(color_binary, M, (imgWidth, imgHeight), flags=cv2.INTER_LINEAR)
    write_binary_image(os.path.join(OUT_DIR, "4_warped_"+image_name+".png"), binary_warped)
    image_height = binary_warped.shape[0]
    image_width = binary_warped.shape[1]
    
    # == FIND LINES ==
    # Detect lane pixels and fit to find the lane boundary.
    # === Histogram ===
    # STARTING POINT WAS DIRECTLY FROM LESSON 14-33. Will modify from there.
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    plt.plot(histogram)
    plt.savefig(os.path.join(OUT_DIR, "5_histogram_"+image_name+".png"))

    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
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

    
    
    # === Vizualizing the lane-finding ===
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, image_width)
    plt.ylim(image_height, 0)
    plt.savefig(os.path.join(OUT_DIR, "6_line_detection"+image_name+".png"))
    plt.close()
    
    
    
    # TODO: MAKE SURE TO SKIP THE SLIDING-WINDOWS STEP ONCE WE HAVE AN IDEA OF WHERE THE LINES ARE (FOR PERFORMANCE REASONS).
    # Some useful code & visualization is in the bottom of 14-33
    
    # NOTE: IF THE SLIDING-WINDOWS DON'T FIT WELL ENOUGH, LESSON 14-34 HAS ANOTHER WAY TO DO IT WHICH MIGHT BE EASIER
    # TO TWEAK INTO SOMETHING THAT WOULD FIT OUR NEEDS.
    
    
    
    # == DETERMINE CURVATURE ==
    # Determine the curvature of the lane and vehicle position with respect to center.
    # Generate some fake data to represent lane-line pixels
    ploty = np.linspace(0, (image_height-1), num=image_height)# to cover same y-range as image
    quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
    # For each y position generate random x position within +/-50 pix
    # of the line base position in each case (x=200 for left, and x=900 for right)
    leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                  for y in ploty])
    rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                    for y in ploty])

    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y


    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, leftx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Plot up the fake data
    mark_size = 3
    plt.plot(leftx, ploty, 'o', color='red', markersize=mark_size)
    plt.plot(rightx, ploty, 'o', color='blue', markersize=mark_size)
    plt.xlim(0, image_width)
    plt.ylim(0, image_height)
    plt.plot(left_fitx, ploty, color='green', linewidth=3)
    plt.plot(right_fitx, ploty, color='green', linewidth=3)
    plt.gca().invert_yaxis() # to visualize as we do the images
    plt.savefig(os.path.join(OUT_DIR, "7_curvature"+image_name+".png"))
    plt.close()
    
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    print("\t\tCurve radii: ", left_curverad, right_curverad)
    # Example values: 1926.74 1908.48
    
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print("\t\tCurve radii in meters: ", left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
   

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
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    plt.imshow(result)
    plt.savefig(os.path.join(OUT_DIR, "8_lane_lines_"+image_name+".png"))
    plt.close()
    
    
    # REMAINING PROJECT STEPS:
    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.



    return image

def write_binary_image(file_name, img):
    """
    Since we use binary images (1's, 0's a lot) but if we save them they look all-black (imwrite seems to
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
for file_index in range(len(files)):
    fullFilePath = os.path.join(IN_DIR, files[file_index])
    
    # All of the image-processing is done in this call
    print("     Processing "+fullFilePath+"...")
    image = mpimg.imread(fullFilePath)
    image = process_image(image, mtx, dist, do_output=True, image_name=files[file_index])
    
    # Take the processed image and save it to the output directory.
    saveFile = os.path.join(OUT_DIR, files[file_index])
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
