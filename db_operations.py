import mysql.connector
import numpy as np
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        """Connect to the MySQL database"""
        try:
            self.connection = mysql.connector.connect(
                host=os.getenv("DB_HOST"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME")
            )
            print("Connected to database successfully")
        except mysql.connector.Error as err:
            print(f"Error connecting to database: {err}")
    
    def close(self):
        """Close the database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Database connection closed")
    
    def add_user(self, name):
        """Add a new user to the database"""
        try:
            cursor = self.connection.cursor()
            query = "INSERT INTO users (name) VALUES (%s)"
            cursor.execute(query, (name,))
            self.connection.commit()
            user_id = cursor.lastrowid
            cursor.close()
            print(f"User {name} added with ID: {user_id}")
            return user_id
        except mysql.connector.Error as err:
            print(f"Error adding user: {err}")
            return None
    
    def add_face_encoding(self, user_id, face_encoding):
        """Add a face encoding for a user"""
        try:
            cursor = self.connection.cursor()
            # Convert numpy array to binary data
            encoding_bytes = pickle.dumps(face_encoding)
            query = "INSERT INTO face_encodings (user_id, encoding) VALUES (%s, %s)"
            cursor.execute(query, (user_id, encoding_bytes))
            self.connection.commit()
            cursor.close()
            print(f"Face encoding added for user ID: {user_id}")
            return True
        except mysql.connector.Error as err:
            print(f"Error adding face encoding: {err}")
            return False
    
    def get_all_face_encodings(self):
        """Get all face encodings from the database"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = """
                SELECT u.id, u.name, f.encoding 
                FROM users u 
                JOIN face_encodings f ON u.id = f.user_id
            """
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            
            # Convert binary data back to numpy arrays
            for result in results:
                result['encoding'] = pickle.loads(result['encoding'])
            
            return results
        except mysql.connector.Error as err:
            print(f"Error getting face encodings: {err}")
            return []
    
    def record_attendance(self, user_id):
        """Record attendance for a user"""
        try:
            cursor = self.connection.cursor()
            query = "INSERT INTO attendance (user_id) VALUES (%s)"
            cursor.execute(query, (user_id,))
            self.connection.commit()
            cursor.close()
            print(f"Attendance recorded for user ID: {user_id}")
            return True
        except mysql.connector.Error as err:
            print(f"Error recording attendance: {err}")
            return False
    
    def get_attendance_records(self):
        """Get all attendance records"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = """
                SELECT a.id, u.name, a.timestamp 
                FROM attendance a 
                JOIN users u ON a.user_id = u.id 
                ORDER BY a.timestamp DESC
            """
            cursor.execute(query)
            results = cursor.fetchall()
            cursor.close()
            return results
        except mysql.connector.Error as err:
            print(f"Error getting attendance records: {err}")
            return [] 