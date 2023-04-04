# COMP 4301 Computer Version Project


---

## Process overview:

1. Proceesing the image => Locate the parking lot. 
2. train the CNN model => use model to detect
---

# Image processing:

## 1. Test image

1. Suppose we have an image of the parking lot that we can use to evaluate the accuracy of our parking lot detection algorithm.
    
    <img width="527" alt="image" src="https://user-images.githubusercontent.com/90260431/229894466-017a6660-82ef-46ba-87e6-6ee9b0859b02.png">

    - This data is a single frame extracted from the video captured by the camera.

## 2. Keep the white and yellow part:

1. As shown in the image, the parking space lines are marked in yellow. Therefore, we aim to preserve the yellow lines while converting all other colors to black and white. To achieve this, we will use white and yellow masks to filter out the background.
    
    <img width="515" alt="image" src="https://user-images.githubusercontent.com/90260431/229894528-516b65af-589b-4470-826a-b8d6789ce1b5.png">
    
    inspired by: 
    
    [How to define a threshold value to detect only green colour objects in an image with Python OpenCV?](https://stackoverflow.com/questions/47483951/how-to-define-a-threshold-value-to-detect-only-green-colour-objects-in-an-image/47483966#47483966)
    
    [How to detect two different colors using `cv2.inRange` in Python-OpenCV?](https://stackoverflow.com/questions/48109650/how-to-detect-two-different-colors-using-cv2-inrange-in-python-opencv)
    

## 3. Convert the image to gray scale

<img width="523" alt="image" src="https://user-images.githubusercontent.com/90260431/229894610-d9941330-3a8e-46c4-a4b7-f83e08b80912.png">

## 4. use openCV Canny method to detect edges:
<img width="524" alt="image" src="https://user-images.githubusercontent.com/90260431/229894817-77dd62de-1513-48bf-a971-5ad1133a20d9.png">


---

