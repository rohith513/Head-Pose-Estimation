# Real-Time Head Pose Estimation using OpenCV and Dlib

This is the Python implementation of Headpose estimation using Dlib's face detector(HOG based) and 6-point facial landmark detector. 

# Requirements

![Python](https://img.shields.io/badge/Python-v3.7-brightgreen) &nbsp;
![OpenCV](https://img.shields.io/badge/OpenCV-v3.4+-brightgreen) &nbsp;
![Dlib](https://img.shields.io/badge/Dlib-v19.21-brightgreen) &nbsp;
![Numpy](https://img.shields.io/badge/Numpy-v1.19-brightgreen)


# Features
- Supports webcam, video and image as input source
- Terminal support for easy execution

## Prerequisites
- [camera_calib](camera_calib.py) is used to calculate intrinsic camera parameters
    - Set the below parameters before running the code 
    

    ```python
    nRows = 9
    nCols = 6
    dimension = 29 #- mm

    workingFolder   = "./snap"
    imageType       = 'jpg'
    ```
- `nRows` and `nCols` are rows and columns of the checkerboard image. `dimension` is the width of each block
- Use [save_images](save_images.py) to capture images required for camera calibration
- Capture multiple images of the checkerboard at different orientations
# Usage

## With webcam

```python
    python headpose.py 
```
### If external webcam is used

```python
    python headpose.py -v 1  
```

## With video file

```dotnetcli
    python headpose.py -v /path/to/video.mp4
```


# Working

- **Face detection:** Dlib's HOG based face detector is used to detect a face with its corresponding bounding box. 
- **Facial landmark detection:** Dlib's 68-point facial landmark detector is used to estimate the location of 68 (x, y)-coordinates that map to facial structures on the face. For faster detection, only 6 facial points are used here.
- **Pose estimation:** Using these 6 facial landmarks, the pose is calculated by OpenCV's PnP algorithm
- The model estimates head pose in 4 directions (Right, Left, Top, Bottom) using Euler angles

### Note:
- 3D world coordinates used here are in some arbitrary reference frame/coordinate system
- If camera calibration is not possible, use approximated intrinsic parameters as shown below

```python
    focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
        
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
```
- Adjust Euler angles based on the intrinsic camera parameters for better sensitivity
- For more accurate facial landmark detection, use more facial landmarks from the below image

[//]:![facial_landmarks_68](samples/facial_landmarks_68.jpg)

&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; <img src="samples/facial_landmarks_68.jpg" width="450" height="400">

&nbsp;

# Output

<img src="samples/poselr.gif" width="400" height="350"> &nbsp; &nbsp; <img src="samples/posetb.gif" width="400" height="350">


# References

- Facial point annotations provided by [IBUG](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
- Camera calibration and 3D model points taken from [here](https://learnopencv.com/camera-calibration-using-opencv/) 
- Dlib's 68-point facial detector can be downloaded from [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)