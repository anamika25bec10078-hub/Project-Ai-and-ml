# Project-Ai-and-ml
[DATABASE]
DB_TYPE = sqlite
DB_NAME = criminal_records.db

[MODEL]
DETECTOR_MODEL = haarcascade_frontalface_default.xml
RECOGNITION_MODEL_PATH = data/trained_model.yml
# For LBPH, a lower number means a better match (distance). Tune this value.
CONFIDENCE_THRESHOLD = 80 

[ALERTS]
EMAIL_SENDER = your.system.email@gmail.com 
EMAIL_PASSWORD = YOUR_APP_PASSWORD_OR_KEY
EMAIL_RECIPIENT = law_enforcement_dispatch@police.com


#code for source code directory
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)

class DatabaseManager:
    """Manages all CRUD operations for the criminal records database (FR1)."""
    
    def __init__(self, db_name='criminal_records.db'):
        self.db_name = db_name
        self.conn = None

    def connect(self):
        """Establishes a connection and creates the table if needed."""
        try:
            self.conn = sqlite3.connect(self.db_name)
            self.create_table()
            logging.info("Database connection established and table checked.")
        except sqlite3.Error as e:
            logging.error(f"Database connection error: {e}")

    def create_table(self):
        """Creates the Criminals table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Criminals (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                crime_details TEXT,
                face_id INTEGER UNIQUE NOT NULL
            )
        """)
        self.conn.commit()

    def get_criminal_by_face_id(self, face_id):
        """Retrieves a criminal record by the recognition FaceID (FR3)."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name, crime_details FROM Criminals WHERE face_id = ?", (face_id,))
        record = cursor.fetchone()
        return record

    def close(self):
        """Closes the database connection (Error Handling)."""
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed.")




            #code for core detection logic
            import cv2
import logging

logging.basicConfig(level=logging.INFO)

class FaceDetector:
    """Handles the detection of faces in an image or video frame."""
    
    def __init__(self, cascade_path='haarcascade_frontalface_default.xml'):
        """Initializes the detector (Correct application of subject concepts)."""
        # Load the Haar Cascade file from the OpenCV data path
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_path)
        if self.detector.empty():
            logging.error("Could not load Haar Cascade classifier XML. Check 'data/' directory.")

    def detect_faces(self, frame):
        """
        Detects faces in a single frame.

        Returns:
            list: A list of bounding boxes (x, y, w, h).
        """
        if self.detector.empty():
            return []

        # Convert to grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30) # NFR3: Robustness against small faces
        )
        
        return faces



        #code for core detection logic
        import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

class FaceRecognizer:
    """
    Handles the training and matching of face features (FR2).
    Uses the Local Binary Patterns Histograms (LBPH) algorithm.
    """
    
    def __init__(self, model_path='data/trained_model.yml'):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.model_path = model_path
        self.load_model()
        
    def load_model(self):
        """Loads the trained recognition model (Validation)."""
        try:
            self.recognizer.read(self.model_path)
            logging.info(f"Loaded recognition model: {self.model_path}")
        except cv2.error:
            logging.warning("No existing trained model found. Training required before recognition.")
            
    def recognize_face(self, face_image):
        """
        Attempts to recognize a face image against the trained model.
            
        Returns:
            tuple: (predicted_id, confidence) or (0, 0) if no model loaded.
        """
        if not self.recognizer:
            return 0, 0

        # Preprocess: Ensure face is grayscale for LBPH
        processed_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        try:
            # Predict returns the predicted ID (face_id) and a confidence value (distance).
            predicted_id, confidence = self.recognizer.predict(processed_face)
            return predicted_id, confidence
        except Exception as e:
            logging.error(f"Recognition failed for face: {e}")
            return 0, 0


            import smtplib
from email.message import EmailMessage
import logging
import configparser

logging.basicConfig(level=logging.INFO)

class AlertService:
    """Handles immediate notification alerts upon criminal identification (FR3)."""
    
    def __init__(self, config_path='config.ini'):
        config = configparser.ConfigParser()
        config.read(config_path)
        # NFR1: Security - using secure connection settings
        self.sender = config['ALERTS']['EMAIL_SENDER']
        self.password = config['ALERTS']['EMAIL_PASSWORD']
        self.recipient = config['ALERTS']['EMAIL_RECIPIENT']

    def send_match_alert(self, criminal_name, crime_details):
        """Sends an urgent email notification."""
        if not self.sender or not self.password or not self.recipient:
            logging.warning("AlertService is not configured. Skipping email alert.")
            return False
            
        try:
            msg = EmailMessage()
            msg.set_content(
                f"URGENT ALERT: Criminal identified in real-time feed.\n\n"
                f"SUSPECT NAME: {criminal_name}\n"
                f"DETAILS: {crime_details}\n"
            )
            msg['Subject'] = f'CRIMINAL MATCH ALERT: {criminal_name}'
            msg['From'] = self.sender
            msg['To'] = self.recipient
            
            # Using SMTP_SSL for a secure connection
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(self.sender, self.password)
                smtp.send_message(msg)
            
            logging.critical(f"Urgent alert successfully sent for: {criminal_name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to send email alert. Error: {e}. Check email settings or app password.")
            return False





            #code for user interface
            import tkinter as tk
from tkinter import ttk
import logging

logging.basicConfig(level=logging.INFO)

class InterfaceGUI:
    """Handles the graphical user interface for the system (NFR3 Usability)."""
    
    def __init__(self, root, start_tracking_callback):
        self.root = root
        self.root.title("Criminal Identification & Tracking System")
        self.start_tracking_callback = start_tracking_callback
        self.status_text = tk.StringVar(value="System Ready.")
        
        self._setup_widgets()
        
    def _setup_widgets(self):
        """Creates the necessary UI elements."""
        
        main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- Control Frame ---
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        
        # Start/Stop Button
        ttk.Button(control_frame, text="Start Real-Time Tracking", command=self.start_tracking_callback).pack(pady=10, fill=tk.X)
        
        # Status/Alert Area
        ttk.Label(control_frame, text="Current Status:", font=("Arial", 10, "bold")).pack(pady=(15, 0))
        ttk.Label(control_frame, textvariable=self.status_text, wraplength=200).pack()

        # Alert Display
        ttk.Label(control_frame, text="--- MATCH ALERT ---", foreground="red", font=("Arial", 12, "bold")).pack(pady=(20, 0))
        self.alert_display = tk.Text(control_frame, height=10, width=30, state=tk.DISABLED)
        self.alert_display.pack()

    def display_alert(self, criminal_name, crime_details):
        """Updates the alert area upon successful identification (FR3)."""
        alert_msg = f"MATCH CONFIRMED!\nSuspect: {criminal_name}\nDetails: {crime_details}"
        self.status_text.set("CRIMINAL IDENTIFIED!")
        
        self.alert_display.config(state=tk.NORMAL)
        self.alert_display.delete('1.0', tk.END)
        self.alert_display.insert(tk.END, alert_msg)
        self.alert_display.config(state=tk.DISABLED)
        logging.critical(f"GUI ALERT: {criminal_name} matched.")

    def update_status(self, msg):
        """Updates the general status message."""
        self.status_text.set(msg)





        #code for system enter entry point and orchestration
        import cv2
import tkinter as tk
import configparser
import logging

# Import all custom modules
from InterfaceGUI import InterfaceGUI
from FaceDetector import FaceDetector
from FaceRecognizer import FaceRecognizer
from DatabaseManager import DatabaseManager
from AlertService import AlertService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CriminalTrackingSystem:
    """
    Main system application. Coordinates Detection, Recognition, DB, and GUI (Proper architectural design).
    """
    def __init__(self, config_file='config.ini'):
        self._load_config(config_file)
        
        # Initialize Core Modules
        self.db_manager = DatabaseManager(db_name=self.config['DATABASE']['DB_NAME'])
        self.db_manager.connect()
        
        # Ensure correct file paths based on the structure
        detector_path = self.config['MODEL']['DETECTOR_MODEL']
        recognizer_path = self.config['MODEL']['RECOGNITION_MODEL_PATH']

        self.detector = FaceDetector(cascade_path=f'data/{detector_path}' if 'data' not in detector_path else detector_path)
        self.recognizer = FaceRecognizer(model_path=recognizer_path)
        self.alert_service = AlertService(config_path=config_file)
        
        self.confidence_threshold = int(self.config['MODEL']['CONFIDENCE_THRESHOLD'])
        
        # Initialize GUI
        self.root = tk.Tk()
        self.gui = InterfaceGUI(self.root, self.start_tracking)
        
        self.video_capture = None
        self.is_tracking = False

    def _load_config(self, config_file):
        """Loads configuration settings (Error Handling)."""
        self.config = configparser.ConfigParser()
        if not self.config.read(config_file):
            raise FileNotFoundError(f"FATAL: {config_file} not found. System cannot start.")

    def start_tracking(self):
        """Starts the video capture and the main processing loop (FR2)."""
        if not self.is_tracking:
            self.video_capture = cv2.VideoCapture(0) # 0 for default camera
            if not self.video_capture.isOpened():
                self.gui.update_status("ERROR: Could not open camera.")
                logging.error("Could not open video camera.")
                return

            self.is_tracking = True
            self.gui.update_status("Tracking ACTIVE...")
            self.process_video_stream()
        
    def process_video_stream(self):
        """The main loop for detection and recognition (NFR2: Performance)."""
        
        if not self.is_tracking:
            return

        ret, frame = self.video_capture.read()
        
        # Check for image integrity (Error Handling)
        if not ret:
            logging.warning("Failed to read frame from camera.")
            self.gui.update_status("Camera Error/Disconnected.")
            self.root.after(30, self.process_video_stream) 
            return

        # 1. Detect Faces
        faces = self.detector.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            
            # 2. Recognize and Validate
            predicted_id, confidence = self.recognizer.recognize_face(face_crop)
            
            # For LBPH, a lower confidence value means a closer match.
            if confidence < self.confidence_threshold: 
                # Match Confirmed! (Validation and error handling)
                criminal_record = self.db_manager.get_criminal_by_face_id(predicted_id)
                
                if criminal_record:
                    name, details = criminal_record
                    
                    # Visual Feedback: Red box for criminal
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3) 
                    cv2.putText(frame, f"CRIMINAL: {name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # 3. Alerting (FR3)
                    self.gui.display_alert(name, details)
                    self.alert_service.send_match_alert(name, details)
                    
            else:
                # Unrecognized
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "UNKNOWN", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame in a separate OpenCV window
        cv2.imshow('Tracking System Feed', frame)
        
        # Use tkinter's after method for controlled loop timing (NFR2)
        self.root.after(30, self.process_video_stream) 

    def run(self):
        """Starts the main GUI application."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
        
    def on_closing(self):
        """Handles cleanup (Error Handling)."""
        self.is_tracking = False
        if self.video_capture and self.video_capture.isOpened():
            self.video_capture.release()
            cv2.destroyAllWindows()
        self.db_manager.close()
        self.root.destroy()
        logging.info("System shut down gracefully.")


if __name__ == "__main__":
    try:
        app = CriminalTrackingSystem()
        app.run()
    except FileNotFoundError as e:
        print(f"\n[FATAL ERROR] {e}. Please ensure config.ini is present.")
    except Exception as e:
        print(f"\n[UNHANDLED ERROR] System encountered a critical error: {e}")



        
