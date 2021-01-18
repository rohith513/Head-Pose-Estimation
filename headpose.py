
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np


#Argument Parsing
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", default='shape_predictor_68_face_landmarks.dat', help="path to facial landmark predictor")
ap.add_argument("-v", "--video", default=0, type=str, help="input source")
args = vars(ap.parse_args())


#Load dlib's face detector(HOG-based)
#Load dlib's 68-point facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


#Initialize input video
try:
    cap = cv2.VideoCapture(int(args["video"]))
except:
    cap = cv2.VideoCapture(args["video"])
    
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('poseresult.mp4',fourcc, 15.0, (int(cap.get(3)),int(cap.get(4))))

if (cap.isOpened()== False):
    print("Error opening video stream or file")

#3D Model coordinates
model_points = np.array([                            
                            (225.0, 170.0, -135.0),      # Right eye right corner
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (0.0, 0.0, 0.0),             # Nose tip  
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0),     # Right mouth corner
                            (0.0, -330.0, -65.0)         # Chin                         
                        ]) 

#Intrinsic Parameters
# camera_matrix = np.array([[1107.66,0,658.18],
#                          [0,1118.04,306.28],
#                          [0,0,1]], dtype = "float")

# dist_cf= np.array([[0.437],[-2.96],[-0.0182],[0.00489],[9.02]])


d = [0,0,0]
j=1


while True:
    ret, frame = cap.read()
    if frame is None:
        break
    #frame = imutils.resize(frame, width=550, height=800)  # resize frame if input source resolution is too big
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape
    
    rects = detector(gray, 0)            # detect faces in grayscale, Use 1 to detect multiple faces 
    
    if len(rects) == 0:
        cv2.putText(frame, 'No faces detected', (50,200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),2)
        
      
    #Loop over face detections
    for rect in rects:
        #Computing and drawing bounding boxes on the frame
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),(0, 255, 0), 1)
        rectcenter = ((bX+bX+bW)/2, (bY+bY+bH)/2)
        
        
        #Intrinsic parameters approximation. Use this if camera is not calibrated.
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
        
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        
        
        #Find and convert facial landmarks to a numpy array
        ref2D = predictor(gray, rect)
        ref2D = face_utils.shape_to_np(ref2D)
        ref2D = np.array(ref2D[[45,36,30,48,54,8],:], dtype = "float")      # select same facial landmarks as 3D model points
        
        
        
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, ref2D, camera_matrix, dist_coeffs)
 

        for p in ref2D:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            
    
        
        rmat, jac = cv2.Rodrigues(rotation_vector)
        
        euangles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
        
        
        cv2.putText(frame, "Top: {:.3f}".format(euangles[0]), (50,200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),2)
        cv2.putText(frame, "Left: {:.3f}".format(euangles[1]), (50,250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),2)
        
          
        d = [a + b for a, b in zip(euangles, d)]
    
        if j%3==0:
            angles = tuple(t/3 for t in d)
            d = [0,0,0]
                    
            if angles[1] < -25 :
                cv2.putText(frame, "Left: {:.3f}".format(angles[1]), (100,320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif angles[1] > 25:
                cv2.putText(frame, "Right: {:.3f}".format(angles[1]), (100,320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif -170<angles[0]<-150:
                cv2.putText(frame, "Bottom: {:.3f}".format(angles[0]), (100,320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            elif 140<angles[0]<170:
                cv2.putText(frame, "Top: {:.3f}".format(angles[0]), (100,320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, 'Forward', (100,320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            
        j+=1
        #time.sleep(0.2)
    out.write(frame)
    cv2.imshow("Frame", frame)
 
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 

cap.release()
out.release()
cv2.destroyAllWindows()
