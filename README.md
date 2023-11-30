#
# Gender Detection & Age Prediction

# Objective :

Create a gender and age detection system capable of estimating the gender and age range of an individual from a given image or webcam feed.

#
# Project Description:

This Python project utilizes Deep Learning techniques to precisely identify gender and age from facial images. The model leverages training from <a href="https://talhassner.github.io/home/projects/Adience/Adience-data.html">Tal Hassner and Gil Levi</a>. Predicted genders are categorized as 'Male' or 'Female,' while predicted age falls into specific ranges: (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), and (60 – 100) [with 8 nodes in the final softmax layer]. The decision to frame the problem as a classification rather than regression accounts for challenges like makeup, lighting, obstructions, and facial expressions that make exact age prediction challenging from a single image.

#
# Dataset :

The Adience dataset is employed for this Python project, publicly accessible <a href="https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification">here</a>. This dataset serves as a benchmark for face photos, encompassing diverse real-world conditions such as noise, lighting, pose, and appearance. Comprising 26,580 photos of 2,284 subjects across eight age ranges, it is distributed under the Creative Commons (CC) license and totals approximately 1GB. The models utilized in this project undergo training using this comprehensive dataset.

# 
# Required Python Libraries:
<ul>
    <li>OpenCV</li>
    
      pip install opencv-python
</ul>
<ul>
    <li>argparse</li>
  
      pip install argparse
</ul>

#
# Project Contents:
<ul>
    <li>opencv_face_detector.pbtxt</li>
    <li>opencv_face_detector_uint8.pb</li>
    <li>age_deploy.prototxt</li>
    <li>age_net.caffemodel</li>
    <li>gender_deploy.prototxt</li>
    <li>gender_net.caffemodel</li>
    <li>A few pictures for testing</li>
    <li>detect.py</li>
</ul>
For face detection, the .pb file is a protobuf file holding the graph definition and trained model weights in binary format. The .pbtxt extension signifies the same content in text format, specifically for TensorFlow. Regarding age and gender, the .prototxt files detail network configuration, while the .caffemodel file specifies internal states of layer parameters.
 
#
# Usage :
 <ul>
    <li>Download the repository.</li>
    <li>Open Command Prompt or Terminal and navigate to the folder containing all files.</li>
    <li><b>To detect gender and age in an image, use the command:</b></li>
  
      python detect.py --image <image_name>
</ul>
    <p><b>Note: </b>The image should be in the same folder as the files.</p> 
<ul>
    <li><b>To detect gender and age through the webcam, use the command:</b></li>
  
      python detect.py
</ul>
<ul>
    <li>Press <b>Ctrl + C</b> to stop program execution.</li>
</ul>

#
# Examples :

    >python detect.py --image girl1.jpg
    Gender: Female
    Age: 25-32 years
    
<img src="Example/Detecting age and gender girl1.png">

    >python detect.py --image girl2.jpg
    Gender: Female
    Age: 8-12 years
    
<img src="Example/Detecting age and gender girl2.png">

    >python detect.py --image kid1.jpg
    Gender: Male
    Age: 4-6 years    
    
<img src="Example/Detecting age and gender kid1.png">

    >python detect.py --image kid2.jpg
    Gender: Female
    Age: 4-6 years  
    
<img src="Example/Detecting age and gender kid2.png">

    >python detect.py --image man1.jpg
    Gender: Male
    Age: 38-43 years
    
<img src="Example/Detecting age and gender man1.png">

    >python detect.py --image man2.jpg
    Gender: Male
    Age: 25-32 years
    
<img src="Example/Detecting age and gender man2.png">

    >python detect.py --image woman1.jpg
    Gender: Female
    Age: 38-43 years
    
<img src="Example/Detecting age and gender woman1.png">
              
#
# Thank You!
<b>Connect with me on Linkedin!!</b>

[![Akshat Singh Linkedin](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/akshtsng/)