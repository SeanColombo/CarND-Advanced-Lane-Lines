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

[calibration_corners]: ./output_images/z1_calibration_with_corners_3.png "Calibration Corners"
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

![Camera Calibration corner detection][calibration_corners]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Camera Calibration undistortion][calibration_undistorted]

All of the successful corner-detection and un-distortions were saved to the `/output_images` directory for reference.

This calibration step took about 10-12 seconds every time I ran the script. Since much of the later development required rapidly seeing the results of small changes, this time was prohibitive to efficient development.  The entire point of this calibration is to obtain the camera-matrix (`mtx`) and distortion-coeffitients (`dist`) to allow un-distorting images later. Therefore, I took those two values and stored them to a pickle-file.  When the pickle-file can be found and loaded correctly, that is used instead of re-doing camera-calibration. This allowed the script to quickly start up and calibrate images without any perceptible delay.

### Pipeline (single images)

The pipeline for processing a single image (or frame of a video) is included in the function `process_image()` defined near line 164 of `[./findLanes.py](./findLanes.py)`.  There are some optional debugging parameters which allowed me to output a very detailed flow step-by-step for each of the static images, while skipping the intermediate-output when processing video-frames. This made it very easy to itereate on the development of the lane-detection, while working with the static images.

#### 1. Provide an example of a distortion-corrected image.

The undistortion is achieved by using the `cv2.undistort()` function and passing in the `mtx` and `dist` values that were obtained from the camera-calibration steps described above.

Interestingly, I managed to do the entire project forgetting to apply the undistortion values to the images in `process_image()`. The entire pipeline worked fine without the undistortion-step but that probably only implies that the camera's distortion is not that extreme.  I have now corrected the pipeline and you can see the small but important effect of undistortion in this example (look near the bottom corners to see the difference):

![undistorted camera image][undistorted]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_video_output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
