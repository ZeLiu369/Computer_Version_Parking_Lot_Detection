# COMP 4301 Computer Version Project


---

## Process overview:

1. Proceesing the image => Locate the parking lot. 
2. train the CNN model => use model to detect
---

# Image processing:

1. Suppose we have an image of the parking lot that we can use to evaluate the accuracy of our parking lot detection algorithm.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9e66d1a4-1ddf-40b9-811c-175f83768580/Untitled.png)
    
    - This data represents a single frame extracted from the video captured by the camera.

## Keep the white and yellow part:

1. As shown in the image, the parking space lines are marked in yellow. Therefore, we aim to preserve the yellow lines while converting all other colors to black and white. To achieve this, we will use white and yellow masks to filter out the background.
    
    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/256fb3a6-69dd-4733-9b24-c4ef92ff2f32/Untitled.png)
    
    inspired by: 
    
    [How to define a threshold value to detect only green colour objects in an image with Python OpenCV?](https://stackoverflow.com/questions/47483951/how-to-define-a-threshold-value-to-detect-only-green-colour-objects-in-an-image/47483966#47483966)
    
    [How to detect two different colors using `cv2.inRange` in Python-OpenCV?](https://stackoverflow.com/questions/48109650/how-to-detect-two-different-colors-using-cv2-inrange-in-python-opencv)
    

## Convert the image to gray scale

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8cec4138-5494-4649-81eb-e9c3c101188a/Untitled.png)

---