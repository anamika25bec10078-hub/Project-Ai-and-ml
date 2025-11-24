# Overview of the implementation

The proposed system is implemented using Python 3.x due
to its extensive support for computer vision libraries and rapid prototyping capabilities.
The architecture is modular, separating the ;ogic for detection, recognition, database management, 
and user interaction into distinct classes, this ensures code reusability and maintainability.

# Module-Wise Logic Explanation

#
1.Face Detection Module(FaceDetector.py)
#

The detection phase serves as the entry point for the pipeline.
# Algorithm:
We utilized the Haar Feature-based Cascade Classifier, a pre-trained model provided
by OpenCV(haarcascade_frontalface_default.xml).
# Logic:
The system captures video frames in real-time.To optimize performance,every captured frame is 
converted from BGR(Blue-Green-Red) to Grayscale.The detectMultiscale function scans the image at 
various scales to identity rectangular regions containing faces.
# Parameter Tuning:
the scaleFactor is set yo 1.1 and minNeighbors to 5 to balance between detection sensitivity and 
reducing false positives(noise).

#
2.Feature Extraction and Recognition Module(FaceRecognizer.py)
#

once a face is detected, the region of interest(ROI) is passed to the recogintion module.
# Algorithm:
The system employs the Local Binary Pattern Histograms(LBPH) algorithm.
Unlike Eigenfaces or Fisherfaces, LBPH is robust against monotonic gray-scale transformations,
making it suitable for varying lighting conditions in real-world surveillance.
# Process:
1. Traing: The model is trained on a dataset of labeled criminal images,learning the local
   structure of face by comparing each pixel with its neighbors.
2. Prediction:When a new face is presented, the system calculates a histogram of the new image
   and compares it with the stores histograms using a distance metric(Confidence score).
3. Thresholding: A confidence threshold (set to 80 in configuration) determines if the
   match is valid. A lower score in LBPH indicates a closer match (less distance/difference).
   
#
3. Database Management(DatabaseManager.py)
#

To ensure persistent storage of criminal records, The system uses SQLite, a serverless, lightweight
database engine.
# Scheme Design:
A single table Criminals is created with columns for id,name,crime_details,and a unique face_id.
# Functionality:
The class implements a Singleton-like pattern for connection handling. It provides methods to fetch 
criminal detils speciafically by their face_id, which links the computer vision output(an integer ID) 
to the semantic data(Name,Crime).

#
4. Alert and notification service(AlertService.py)
#

Upon a successful identification (where confidence<threshold), the system triggers an asynchronous 
alert.
# Implementation:
The smtplib library is used to establish a secure SSl connection with an SMTP serve(e.g., Gmail).
# Logic:
An EmailMessage object is constructed containing the identified suspect's name and crime details.
This ensures that law enforcement is notified even if they are monitoring the screen remotely.

#
5. Main Control Loop(Main.py)
#
The central orchestration occurs in the main execution loop.
# Integration:
It initializes the GUI(Tkinter) and the Video Capture object.
# Workflow:
The loop captures a frame--> calls FaceDetector--> passes results to FaceRecognizer--> 
queries DatabaseManager if a match is found--> updates InterfaceGUI.
# Latency Management:
The loop uses the root.after() method rather than a while True loop to prevent freezing the
Graphical User Interface(GUI), ensuring the application remains responsive during video processing.


# Libraries and Tools Used

Library           Purpose
OpenCV(cv2)       Core image processing,Haar Cascade detection, and LBPH recognition.
Tkinter           Building the user interface (GUI) for the tracking dashboard.
SQLite3           SQL database implementation for storing suspect records.
Numpy             Handling image arrays and numerical matrix operations required by OpenCV.
SMTPlib           Handling the transmission of email alerts via TCP/IP.


