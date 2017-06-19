**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[calibration_undistorted]: ./output_images/z2_undistorted_chess_board_3.png "Calibration Undistorted"
[undistorted]: ./output_images/2_undistorted_straight_lines1.png "Undistorted camera image"
[project_video_output]: ./test_videos_output/project_video.mp4 "Project Video Output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the section of code with the title **# == CAMERA CALIBRATION ==** in `findLanes.py` around line 436.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I assumed that all images would have 9 corners detected horizontally and 6 detected vertically. Most images were able to find all of these corners, but some calibration images had too much of the image cropped off so not all of the corners could be found. To prevent these images from potentially confusing the calibration, those images were not used for calibration. Of the 20 calibration images, only 3 (image 1, image 4, and image 5) could not find a good match - the other 17 images were used.  Here is an example of the results of corner-detection that `cv2.findChessboardCorners()` provides:

<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Advanced-Lane-Lines/master/output_images/z1_calibration_with_corners_3.png" width="350">

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Camera Calibration undistortion][calibration_undistorted]

All of the successful corner-detection and un-distortions were saved to the `/output_images` directory for reference.

This calibration step took about 10-12 seconds every time I ran the script. Since much of the later development required rapidly seeing the results of small changes, this time was prohibitive to efficient development.  The entire point of this calibration is to obtain the camera-matrix (`mtx`) and distortion-coeffitients (`dist`) to allow un-distorting images later. Therefore, I took those two values and stored them to a pickle-file.  When the pickle-file can be found and loaded correctly, that is used instead of re-doing camera-calibration. This allowed the script to quickly start up and calibrate images without any perceptible delay.

### Pipeline (single images)

The pipeline for processing a single image (or frame of a video) is included in the function `process_image()` defined near line 164 of `[./findLanes.py](./findLanes.py)`.  There are some optional debugging parameters which allowed me to output a very detailed flow step-by-step for each of the static images, while skipping the intermediate-output when processing video-frames. This made it very easy to itereate on the development of the lane-detection, while working with the static images.

#### 1. Provide an example of a distortion-corrected image.

The undistortion is achieved by using the `cv2.undistort()` function and passing in the `mtx` and `dist` values that were obtained from the camera-calibration steps described above.

Interestingly, I managed to initially do the entire project forgetting to apply the undistortion values to the images in `process_image()`. The entire pipeline worked fine without the undistortion-step but that probably only implies that the camera's distortion is not that extreme.  I have now corrected the pipeline and you can see the small but important effect of undistortion in this example (look near the bottom corners to see the difference):

![undistorted camera image][undistorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Color transforms and gradients were easily encapsulated, so immediately after the undistortion in `process_images()`, I made a call out to `color_gradient_pipeline()` (defined around line 116 of `findLanes.py`) which includes all of the code for color transforms and gradients.

There were a bunch of steps in here, and I created images for all of the static test-images at each step in this process.  Here is an example of all of the steps for one test-image:

Converting image to HLS color-space:

<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Advanced-Lane-Lines/master/output_images/3.10-hls-straight_lines1.png" width="350">
Using Sobel operator on X:

<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Advanced-Lane-Lines/master/output_images/3.20-sobel-straight_lines1.png" width="350">
Threshold X gradient (magnitude of the gradient):

<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Advanced-Lane-Lines/master/output_images/3.30-thresh_x_gradient-straight_lines1.png" width="350">
Separating out the S-Channel:

<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Advanced-Lane-Lines/master/output_images/3.40-s_channel-straight_lines1.png" width="350">
Threshold on the S-Channel to make binary output:

<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Advanced-Lane-Lines/master/output_images/3.41-s_binary-straight_lines1.png" width="350">
Stacking the thresholding/gradients and s-channel values to show what they each contribute:

<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Advanced-Lane-Lines/master/output_images/3.45-color_binary-straight_lines1.png" width="350">
Combined the binary values that are shown in the prior steps:

<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Advanced-Lane-Lines/master/output_images/3.50-stacked_binaries-straight_lines1.png" width="350">

Through this process, we took a full-color image and returned a binary image which has a pretty good starting point for "seeing" lines of all colors (white and yellow in our use-case) even in varying light conditions (eg: shadows). This binary image will be used for the remainder of the processing in the pipeline.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

After the call to `color_gradient_pipeline()` (starting near line 203 of `findLanes.py`), it was necessary to perform a perspective-transform on the binary image to make it so that we were in effect "looking down" onto the road.

This is accomplished by creating a trapezoidal "source" area and transforming/warping it into a rectangular destination.

As an aside: this step was a great example of why it is very useful to visually debug every step in your pipeline. My early results of the whole pipeline were quite awful and it was all because I use the source-trapezoid that I had created in Project 1 and did not initially debug its appropriateness for this particular use-case.  Once I did start outputting an overlay of the source-trapezoid onto the input images, it became clear that it was far from what I needed.  Here is the initial source-area:

<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Advanced-Lane-Lines/master/output_images/z_Original-Area-of-Interest.png" width="350">

The problems with this area-of-interest are that with the way this camera is mounted (as opposed to Project 1 where this region was created for) it is way to "short" and wide. There is very little room to see the lane ahead, and a lot of wasted space in the shoulder & off in the next lane. The short lane gives insufficient information to see the dashed white lines while the side-information could confuse the lane-detection when it sees things like changes in pavement, jersey-barriers, etc..

Here is a view of a more appropriate area-of-interest that I found worked much better for this project:
<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Advanced-Lane-Lines/master/output_images/4.0-pre-warp-areaOfInterest-straight_lines1.png" width="350">

When choosing this area, it was important to make sure that the "side" lines of the trapezoid were fairly close to parallel to the lane-liens in the "straight" sample images.  This leads to the perspective-transform being what we need. One way that I tweaked this was to change the trapezoid slightly, then see how it affected the "transformed" images. When the perspective-transformed images for the two straight-line test images had what appeared to be parallel, verticle lines, then the calibration was complete.  I had variables in my code (primarily `trapazoidTopWidth` which is the width of the top-line of the trapezoid, which results in controlling the angles of the side-lines of the trapezoid) that allowed me to tweak this transformation with sufficient precision.

The "destination" that I projected to, was a simple rectangle filling the image.

The code to create source/dest from a few configuration variables (just some trig) is here:

```python
    trapazoidTopWidth = imgWidth * 0.10 # this is the width of the top line of the trapazoid
    trapazoidHeight = imgHeight * 0.32 # guess/test/revised to tune this number (this will be actual height of trapezoid)
    PADDING_FROM_SIDES = 50 # how many pixels from the side of camera before the left side of the trapazoid starts
    PADDING_FROM_BOTTOM = 40 # how many pixels to shave off the bottom of the image (basically, the hood of the car - just measured it in Gimp)

    # Trig to figure out the points in the trapezoid based on the configuration & image size:
    xOffset = (trapazoidTopWidth / 2) # distance that trapezoid top points will be from vertical center-line
    theta = math.atan( trapazoidHeight / (((imgWidth/2)-xOffset)-PADDING_FROM_SIDES) )
    topLeftX = ( (imgWidth/2) - xOffset )
    topLeftY = imgHeight - PADDING_FROM_BOTTOM - trapazoidHeight
    topRightX = ( (imgWidth/2) + xOffset )
    topRightY = topLeftY
    src = np.array([[
        (topLeftX, topLeftY), # top left
        (topRightX, topRightY), # top right
        (imgWidth-PADDING_FROM_SIDES, imgHeight-PADDING_FROM_BOTTOM), # bottom right
        (PADDING_FROM_SIDES, imgHeight-PADDING_FROM_BOTTOM) # bottom left
    ]], dtype=np.float32)

    PADDING = 0 # it turns out that no padding was necessary. The lessons appeared to have horizontal padding but I couldn't think of any value that would provide.
    dst = np.float32([[PADDING, PADDING], # top left
                      [imgWidth-PADDING, PADDING], # top right
                      [imgWidth-PADDING, imgHeight-PADDING], # bottom right
                      [PADDING, imgHeight-PADDING]]) # bottom left
```

These source and destination values are passed into `cv2.getPerspectiveTransform()` to obtain the transformation matrix that can then be passed into `cv2.warpPerspective()` to get the top-down view of the image.

Here is a perspective-transformed version of the binary source above. You'll note that since the "source" region fairly closely matches the area of a lane, that much of the un-desired information in the image (trees, cars, etc.) no longer appears in our data.  Also: the lines have been projected to a relatively parallel position.

<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Advanced-Lane-Lines/master/output_images/4.1-warped-straight_lines1.png" width="350">

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Now that the input images have been HEAVILY processed (thresholded, gradiented, perspective-transformed), the actual lane-finding could begin. Starting around line 265 of `findLanes.py`, I used a Histogram to identify the pixel-heavy areas near the bottom of the bitmap. This gave a good starting point for the sliding-widnow search since we can be fairly confident that the car should be able to at least see the lines close to the hood of the car even if the lane curves significantly.

This histogram (showing the location of white-lines near the bottom of the prior image) has peaks in the areas where we should start the sliding-window search:

<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Advanced-Lane-Lines/master/output_images/5_histogram_straight_lines1.png" width="350">

Once those high-points were found, that was used as the starting location for a sliding window search to find the lane lines. I used 9 stacked windows which were each 200 pixels wide (100px to each side of the center of the window).

The points found in these sliding windows were then passed into numpy's `polyfit()` function to use the least-squares method to fit a curve to the points.  The 9 windows, the points in each window, and the left & right lane lines (yellow) can be seen in this image:

<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Advanced-Lane-Lines/master/output_images/6_line_detectionstraight_lines1.png" width="350">

Originally, there were occasional frames that were fairly erroneous due to extreme shadows, road height changes on bridges, etc.. To avoid this situation, we used `cv2.matchShapes()` and discarded any lane-detections that were found to be significantly different from the prior frame. In these cases, we continue to use our prior polygons until a more reasonable delta is obtained. This has a "smoothing" effect which removes very messed up frames and will also have a more smoothed-out assumption in cases where the car is bouncing up and down after going on and off of the bridges in the project video.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Starting around line 359 of `findLanes.py`, I calculated the radius of curvature of the lane using the [radius-of-curvature forumula](http://www.intmath.com/applications-differentiation/8-radius-curvature.php). This resulted in a pixel-space number which I then converted to world-space using the hints provided for this camera:
```python
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
```

The curvature is a huge radius on relatively straight lanes and smaller on very curvy portions of the road.

In addition to the curvature, the car's distance from the center of the lane also had to be calculated. Originally, I made the error of calculating the distance from the center of the **image** but that was obviously incorrect. Now the center of the lane is calculated from the 10th-from-bottom-row of the green polygon, and compared to the center of the car.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Starting around line 385 of `findLanes.py`, I warped the detected lane-lines back from the top-down perspective to the normal camera perspective so that I could visualize the efficacy of the lane-detection.  This uses the `cv2.warpPerspective()` function again, but uses the `Minv` transformation matrix which is the INVERSE of the orignal transformation-matrix that was used to obtain the top-down view. The inverse transformation matrix is obtained by calling `cv2.getPerspectiveTransform()` with the source and destination values inverted.

In addition to rendering this area onto the undistorted versions of the test-images, I also applied the required annotations for the radius of curvature, and the estimate of how far (in world-space) the center of the car was from the center of the lane (negative numbers indicate that the car is left-of-center, positive numbers indicate that the car is right-of-center).

<img src="https://raw.githubusercontent.com/SeanColombo/CarND-Advanced-Lane-Lines/master/output_images/straight_lines1.jpg" width="350">

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_video_output/project_video.mp4)

The detection performs quite well. The only visible shortcoming is that the lane detected is a bit narrow at the top in some cases. This isn't a large problem, but if we desired better accuracy there are some tweaks that could be made to improve that issue (and several other areas for improvement) as described below.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In addition to the debug images above, in the `output_images` directory, there are images for ALL of the static test images which correspond to the images used in this writeup. Having this diverse test-set easily visible during development did a great job of highlighting any weaknesses or bugs very early on so that I didn't go down the wrong path very often.

Each segment above was accompanied by descriptions of areas where I struggled (eg: the initial Area of Interest being way too short and wide) and things I completely messed up (forgetting to unwarp the images until after I had the entire pipeline working), but there are things that I did NOT yet do, that could still be done to improve this project.

While making the project, I kept a running list of all of the things I absolutely needed to do, as well as improvements sorted by their expected level of impact. Then I just kept improving it until the results exceeded the requirements.  Here are some remaining areas for improvement:

* Could add a sanity-check to make sure the lane-lines are roughly parallel. This would address the aforemention weakness that sometimes the very top of the lane-lines are too narrow at the top. Since we know that the left-lane is solid-yellow and easier to detect in this use-case, we could make the right-lane-line bias to more closely match the left, but this solution would be very project-specific and would not extend well to the general use-case of lane detection.  In the more general case, we could take an average of the two lanes curvatures if both lanes passed the OTHER sanity-check mentioned below.
* Could add a santiy-check to make sure the curvatures are realistic based on the [US specifications for highway curvatures](http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm#BGBHGEGC). If one line did not fall within reasonable bounds but the other did, the bad line could just be replaced by an average of the previously-known line and a parallel line to the reasonable line of the current image.
* The image-processing took over 2 minutes on commodity hardware for a 50 second video. It's unclear what kind of hardware would be running this code but it is highly likely that this would be too slow of an implementation for real deployment (which needs to be realtime (eg: take less than 50 seconds to process a 50 second video) and ideally shouldn't use the whole processor since there is lots of other work for a Self Driving Car to do at the same time as lane-detection.  To improve this issue we could write code to optimize lane-finding in sequential images (eg: use a Line class to remember lines from previous images) then instead of doing the sliding-window search, we could always start the search near previously-seen lines. This approach should make the lane-detection much faster.
* The lane-curvature calculation worked fine in practice but I started another implementation that can be found in the  `curvatures_via_convolutions()` method.  Frankly, I didn't expect my initial implementation to be sufficient, but it turned out great so I didn't need to fully work through the second approach!  The spot to substitute it in, is in the code, commented out near line 357 of `findLanes.py`

