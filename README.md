[![Vehicle Detection & Tracking - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
## Dean Webb - Vehicle Detection & Tracking  Pipeline
#### Self-Driving Car Engineer Nanodegee - Project 5
In this project, my goal is to write a software pipeline to extract features from a Dataset and identify the lane boundaries within an input video. In this project, my ultimate goal was to write a software pipeline that allows for ***minimal tuning of hyperparameters*** (e.g. a linear combination of various scaled windows) and automatically extract bounding boxes from the images. The implementation would ideally detect vehicles in a video (e.g. the test_video.mp4 and the full project_video.mp4).

To satify the project requirements, I implemented tracking using combination of 1) Sliding Windows, 2) Hog sub-sampling, 3) False-positive filtering, and 4) Re-using found vehicles (*See e.g., code cells containing functions* **`single_img_features`, `draw_single_frame_labeled_bboxes`,** and **`draw_multi_frame_labeled_bboxes`** in iPython notebook `vehicle-detection-setup.ipynb`)

Fortunately, I was also able to install and configure the YOLO library ("You only look once") To detect vehicles! It is not optimized yet (I'm currently processing the frames by loading the weights for each frame one at a time, but it was a helpful project for learning. The project goals are listed in detail below.

---

![alt text][image8]

---

### <font color='green'>Project Goals</font>

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---

### Dependencies

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

Some example images for testing your pipeline on single frames are located in the `test_images` folder.  To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called `ouput_images`, and include them in your writeup for the project by describing what each image shows.    The video called `project_video.mp4` is the video your pipeline should work well on.

---

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: examples/sliding_windows_code_1.jpg
[image4]: examples/sliding_window.png
[image5]: examples/bboxes_and_heat.jpg
[image6]: examples/udacity_image_augment_1.jpg
[image7]: examples/udacity_image_augment_2.jpg
[image8]: examples/yolo_predicts.jpg
[video1]: project_output_yolo.mp4
[video2]: test_output_yolo.mp4
[image10]: examples/project_car.jpg
[image11]: examples/project_noncar.jpg
[image12]: examples/udacity_car.jpg
[image13]: examples/udacity_noncar.jpg
[image14]: examples/HOG_example_project.jpg
[image15]: examples/HOG_example_udacity.jpg
[image16]: examples/SVC_Classifier.jpg
[image17]: examples/sliding_windows_code_2.jpg
[image18]: examples/search_windows_code.jpg
[image19]: examples/draw_multiframe_labels_code_1.jpg
[image20]: examples/draw_multiframe_labels_code_2.jpg
[image21]: examples/draw_multiframe_labels_code_3.jpg
[image22]: examples/sliding_windows_grid_10_thresh.jpg
[image23]: examples/sliding_windows_grid_fire.jpg
[image24]: examples/sliding_windows_grid.jpg
[image25]: examples/yolo_code.jpg



## <font color='red'> Rubric Points</font>
### Here I will consider the [rubric](https://review.udacity.com/#!/rubrics/513/view) points individually and describe how I addressed each point in my implementation below.

---

### Writeup / README

#### 1. <font color='green'>Provide a Writeup / README</font> that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

**Done!** - *See below.*


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in at least code cell 12 of the accompanying IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images. Optionally, I include a flag that augments the given dataset with the [open-sourced Udacity dataset](http://bit.ly/udacity-annoations-crowdai) (labeled from CrowdAI). To accomplish this, the pipeline downloads and extracts images from the source site, then uses the labeled bounding boxes to extract out the car images, separated by the data set.

It then also takes a snapshot above or to the bottom-left of the car image to balanace out the dataset with a `non-car` image. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

|  Project Set Car  |Project Set NonCar|  Udacity Set Car  |Udacity Set Non-Car|
|:-----------------:|:----------------:|:-----------------:|:-----------------:|
|    ![][image10]   |   ![][image11]   |   ![][image12]    |    ![][image13]   |


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the **`YCrCb`** color space and `HOG` parameters of **`orientations=9`, `pixels_per_cell=(8, 8)`,** and **`cells_per_block=(2, 2)`**:

|  Project Set HOG Features  |
|:--------------------------:|
|        ![][image14]        |

Below is a similar set of examples, but with the Udacity Dataset used for image augmentation.

|  Udacity Set HOG Features  |
|:--------------------------:|
|        ![][image15]        |

---

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and feature extraction techniques. Intiailly, I tried to stack the `color_histogram`, `spatial_binning`, and all channels of the `hog` technique in a feature vecture but I found the `spatial_binning` features made very many false positives when I processed the video.

I also experimented with smoothing and with many different scaling parameters, with the goal in mind to feed in as many windows as possible, then let my Vehicle class filter through the false positives. This ended up working pretty well, but there were too many false positives still. On the bad side, it took way too long to process.

As it turns out there was a bug (it always does it seems) in my feature extraction algorithm, I realized I was converting my images to YCrCb when predicting, but I forgot to do the converting before feeding my classifier. Once I fixed the color space conversion issue, the false positives went away. In fact, it seemed to work well enough without the `spatial_binning` features, which I removed during my attempts to debug the false positives. In this regard, I utilize the `color_histogram` features after converting to `HSV`, and use the `hog` features with a) 9 orientations, b) 8 Pixels per cell, and c) 2 cells per block. The `hog` orientations works pretty well with more than 6, but I found 9 to work the best.

Additionally, I used scaling parameters to resize the images before running through the classifier. Below is an example the scaling parameters I used:

```sh
## Parameters - HOG Sub-Sampling ##
SW_YSTART = 400
SW_YSTOP = 656
SW_SCALES = [1.0, 1.5, 1.75]
SW_CONVERT_COLOR = 'RGB2YCrCb'
```

---

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the features listed above. Here's an example of the output in the attached iPython Notebook of the SVM classifier results. (98.85%)

|SVM Classifier Code Snippet |
|:--------------------------:|
|        ![][image16]        |

One issue that I noticed is that the classifier appears to decrease I a little the more data from the Udacity set that I add. I believe this is normal behavior since I can't be sure that my non-car features did not have some car images accidentally added in (since I added all images automatically).

---

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image. As I noted above, this experiment came with mixed results due to my earlier noted bug. Although I used HOG subsampling, I also included the sliding windows technique in my final algorithm sort of as a redundancy. Below are some parameters passed into my **`sliding_windows`**:

```sh
## Sliding Windows Parameters ##
SW_XSTART_STOPS = [(200, None), (256, 1000)]
SW_YSTART_STOPS = [(384, 640), (384, None)]
SW_XY_WINDOWS = [(96,96),(128,128)]
SW_XY_OVERLAPS = [(.450,.480),(.21,.280)]
```

Here's the **`sliding_windows`** helper functions I used to take in a list of scaling parameters:

|Sliding Windows Code Snippet|
|:--------------------------:|
|        ![][image3]         |
|        ![][image17]        |

I then used a **`search_windows`** helper function to loop through and check if the given windows contained a car image:

|Search Windows Code Snippet |
|:--------------------------:|
|        ![][image18]        |

---

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As mentioned above, I implemented a combination of 1) Sliding Windows, 2) Hog sub-sampling, 3) False-positive filtering, and 4) Re-using found vehicles (*See e.g., code cells containing functions* **`single_img_features`, `draw_single_frame_labeled_bboxes`,** and **`draw_multi_frame_labeled_bboxes`** in iPython notebook `vehicle-detection-setup.ipynb`). I searched on multiple scales using a conversion to `YCrCb` (and all 3-channels of `HOG` features plus color histograms in the `HSV` colorspace) The combination of these techniques are then wrapped in a flag to turn them on and off, which provided a nice way of testing their influence to the final bounding boxes. Here's a code snippet of the pipeline processing a video frame being:

|Video Pipeline Code Snippet |
|:--------------------------:|
|        ![][image19]        |

Here are some example images that were output (in this case from the function **`draw_single_frame_labeled_bboxes`**):

|Single Frame Example Images |
|:--------------------------:|
|        ![][image4]         |

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](project_output_no_yolo.mp4). Here's the video embdedded:


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used **`scipy.ndimage.measurements.label()`** to identify individual blobs in the heatmap. Since I ultimately decided to apply multiple techniques at the same time, I created a list of labels from each respective technique, then merged the results together. I then assumed each remaining blob in the heatmap corresponded to a detected vehicle, so I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of **`scipy.ndimage.measurements.label()`** and the bounding boxes then overlaid on the last frame of video. Here are example frames and their corresponding heatmaps:

|Heatmap and Labels Examples |
|:--------------------------:|
|        ![][image5]         |


Note that the the resulting bounding boxes are drawn onto the last frame in the series.

---

### <font color='red'>Discussion / Learnings / Shortcomings</font>

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

At first, due to errors in my implementation, I had much trouble getting the SVM classifer to predict much of anything correctly. This I found out much later was because I forgot to convert the image to 'YCrCb' before extrracting the `hog` features, although I performed the conversion right before prediction.

As a way to increase the accuracy (and decrease false positives), I include a flag that augments the given dataset with the [open-sourced Udacity dataset](http://bit.ly/udacity-annoations-crowdai) (labeled from CrowdAI). To accomplish this, the pipeline downloads and extracts images from the source site, then uses the labeled bounding boxes to extract out the car images, separated by the data set. Some code snippets are printed below:

|Udacity Dataset Augmentation|
|:--------------------------:|
|        ![][image6]         |
|        ![][image7]         |

As noted above, my combination of 1) Sliding Windows, 2) Hog sub-sampling, 3) False-positive filtering, and 4) Re-using found vehicles (*See e.g., the code cells containing functions* **`single_img_features`, `draw_single_frame_labeled_bboxes`,** and **`draw_multi_frame_labeled_bboxes`** in iPython notebook `vehicle-detection-setup.ipynb`) seems to work pretty well.

For extra practice and added redundancy, I also implemented **YOLO** to investigate/compare how well this network processed the images compared to my original algorithm.

---

##### <font color='green'>Learnings</font>
For the HOG-subsampling and sliding windows techniques, I used a somewhat limited region to search, this was to limit the false positives from the trees or the roads. Additionally, I tried to combat false positives with an image augmentation approach that utilizes the Udacity dataset. I did this by creating a bounding box to the ***lower left*** (to mimic roads) or ***directly above*** (to mimic skies) the ground truth labels.

Here's some example images that shows how much of a disaster the prediction windows were at first:

| Sliding Windows False Postives  |
|:-------------------------------:|
|           ![][image22]          |
|           ![][image23]          |
|           ![][image24]          |

---

##### <font color='red'>Shortcomings</font>
The biggest issue I noticed was the bounding box not always bounding the entire car image within a box. I believe this is due to the smaller windows in my implementation tending to pick up the image first. I implement a matching schemed on the boxes, which causes the b=size of the box to gradually rise. Along those lines, the boxes disappear once the car starts to disappear fromt he frame. To fix the latter issue, I could augment my dataset with more pictures of car images that are halved or warped by some other means.

I could have spent another few weeks on this project perfecting it. One optimization I had considered was to completely train the YOLO network on my augmented Udacity dataset (discussed above). This I believe will help with **YOLO** dropping its detection from time to time. Below is a code snippet on how I processed the frames with YOLO:

|YOLO Code Snippet |
|:----------------:|
|   ![][image25]   |

I ran out of time, but my next experiment was going to be to hopefully use the bounding boxes predicted from **YOLO** as it's own feature extractor somehow to make the image detection more robust and less error prone. My thinking here is I could extract out the bounding boxes using the bounding box color as a pixel value to search for. Once I extracted the boxes, I could integrate them with my `Vehicle()` class for storing and reusing.
