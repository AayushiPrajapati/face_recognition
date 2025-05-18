import cv2
import face_recognition
import numpy as np
from db_operations import DatabaseManager

class FaceRecognitionSystem:
    def __init__(self):
        self.db = DatabaseManager()
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.load_known_faces()
        
    def load_known_faces(self):
        """Load known faces from the database"""
        face_data = self.db.get_all_face_encodings()
        
        self.known_face_encodings = [data['encoding'] for data in face_data]
        self.known_face_names = [data['name'] for data in face_data]
        self.known_face_ids = [data['id'] for data in face_data]
        
        print(f"Loaded {len(self.known_face_encodings)} face(s) from database")
    
    def register_new_face(self, name, face_image):
        """Register a new face in the system"""
        # Convert image to RGB (face_recognition uses RGB)
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_image)
        
        if not face_locations:
            return False, "No face detected in the image"
        
        if len(face_locations) > 1:
            return False, "Multiple faces detected. Please provide an image with only one face"
        
        face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
        
        # Add user to database
        user_id = self.db.add_user(name)
        if user_id:
            # Add face encoding to database
            success = self.db.add_face_encoding(user_id, face_encoding)
            if success:
                # Update local lists
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)
                self.known_face_ids.append(user_id)
                return True, f"User {name} registered successfully"
        
        return False, "Failed to register user"
    
    def recognize_faces(self, frame):
        """Recognize faces in a frame"""
        # Convert frame to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find face locations and encodings
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        face_names = []
        recognized_ids = []
        
        for face_encoding in face_encodings:
            # Compare with known face encodings
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            user_id = None
            
            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
                    user_id = self.known_face_ids[best_match_index]
                    
                    # Record attendance
                    self.db.record_attendance(user_id)
            
            face_names.append(name)
            recognized_ids.append(user_id)
        
        return face_locations, face_names, recognized_ids
    
    def close(self):
        """Close database connection"""
        self.db.close() 