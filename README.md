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

## 5. Crop Parking lot spaces

![image](https://github.com/ZaneLeo111/4301_Computer_Version_Parking_Lot_Detection/assets/90260431/bdd0091e-74d9-4025-bc0a-dd100810fac2)

We crop out any parts of the image that are not related to the actual parking lot. For more accurate and precise calculations and predictions down the line.
## 6. Hough line transform:
![image](https://github.com/ZaneLeo111/4301_Computer_Version_Parking_Lot_Detection/assets/90260431/c59c882c-04a4-4858-adde-3e5addf3f58c)

At this point we now will use hough line transform to find straight lines within our previously detected edges to find and locate likely spots for parking.

## 7. Identify rectangular blocks of parking:
![image](https://github.com/ZaneLeo111/4301_Computer_Version_Parking_Lot_Detection/assets/90260431/018c124d-b00b-4729-a43f-70b6f3f33cdd)
Here we pretty much use what we found from the last step to mark areas for parking blocks. This gives us a more focused area to specifically work on from this step forward.

## 8.Identify each spot and count num of parking spaces:

![image](https://github.com/ZaneLeo111/4301_Computer_Version_Parking_Lot_Detection/assets/90260431/8c2fd70a-fba0-4690-8113-28e3be81839f)

Here we mark and identify the different parking spots found within the lot based on the width of the lines we previously found within the areas we defined. Here like many spots in the code we opted to visualize an output for testing purposes and to better explain the methods.

## 9. Crop spot images for CNN model (using code and data pre-processing steps before), and train the CNN model

![image](https://github.com/ZaneLeo111/4301_Computer_Version_Parking_Lot_Detection/assets/90260431/6d85429d-a143-4f3a-9b76-7f7d6e855a61)

![image](https://github.com/ZaneLeo111/4301_Computer_Version_Parking_Lot_Detection/assets/90260431/eabb69f6-3e6d-4fa4-8b20-050589dd62b7)
### CNN parameters:
```
We build on the top of VGG16 with some hyper-parameters:
batch_size = 32
epochs = 15
activation = softmax
optimizers.SGD(lr=0.0001, momentum=0.9)
```

The model training and testing loss:  

![image](https://github.com/ZaneLeo111/4301_Computer_Version_Parking_Lot_Detection/assets/90260431/325bbbf8-18e2-43ec-b907-b4f2fc631b8e)




## 10. After all our different methods being executed we finally get our final result which predicts and highlights all the available parking spots as well as the total amount of parking spots within the lot that the neural network found.
![image](https://github.com/ZaneLeo111/4301_Computer_Version_Parking_Lot_Detection/assets/90260431/93ebf840-72a7-48dc-a5e2-dcd1e0dec868)

For every frame of the video, used for result testing. 


