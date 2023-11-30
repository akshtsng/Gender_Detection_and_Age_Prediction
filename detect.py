# A Gender and Age Detection program by Akshat Singh

# Import necessary libraries
import cv2
import math
import argparse

# Function to highlight faces in the image
def highlightFace(net, frame, conf_threshold=0.7):
    # Copy the frame
    frameOpencvDnn=frame.copy()
    # Get frame height and width
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    # Create a blob from the frame
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    # Set the blob as input to the network
    net.setInput(blob)
    # Perform a forward pass of the network
    detections=net.forward()
    faceBoxes=[]
    # Loop over the detections
    for i in range(detections.shape[2]):
        # Get the confidence of the detection
        confidence=detections[0,0,i,2]
        # If confidence is greater than the threshold
        if confidence>conf_threshold:
            # Calculate the coordinates of the bounding box
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            # Append the bounding box coordinates to faceBoxes
            faceBoxes.append([x1,y1,x2,y2])
            # Draw the bounding box on the frame
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    # Return the frame and the bounding boxes
    return frameOpencvDnn,faceBoxes

# Create an argument parser
parser=argparse.ArgumentParser()
# Add an argument for the image
parser.add_argument('--image')

# Parse the arguments
args=parser.parse_args()

# Define the paths for the models and prototxt files
faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

# Define the mean values for the model
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
# Define the age groups
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# Define the genders
genderList=['Male','Female']

# Load the face detection model
faceNet=cv2.dnn.readNet(faceModel,faceProto)
# Load the age detection model
ageNet=cv2.dnn.readNet(ageModel,ageProto)
# Load the gender detection model
genderNet=cv2.dnn.readNet(genderModel,genderProto)

# Open the video file or webcam
video=cv2.VideoCapture(args.image if args.image else 0)
# Define the padding around the face for face detection
padding=20
# Loop over the frames from the video
while cv2.waitKey(1)<0 :
    # Read a frame from the video
    hasFrame,frame=video.read()
    # If no frame is read, then break the loop
    if not hasFrame:
        cv2.waitKey()
        break
    
    # Highlight the faces in the frame
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    # If no face is detected, print a message
    if not faceBoxes:
        print("No face detected")

    # Loop over the detected faces
    for faceBox in faceBoxes:
        # Extract the face from the frame
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        # Create a blob from the face
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        # Set the blob as input to the gender detection model
        genderNet.setInput(blob)
        # Perform a forward pass to detect the gender
        genderPreds=genderNet.forward()
        # Get the detected gender
        gender=genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        # Set the blob as input to the age detection model
        ageNet.setInput(blob)
        # Perform a forward pass to detect the age
        agePreds=ageNet.forward()
        # Get the detected age
        age=ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        # Draw the detected age and gender on the frame
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        # Display the frame
        cv2.imshow("Detecting age and gender", resultImg)
