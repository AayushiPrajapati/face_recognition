import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import platform
from face_recognition_module import FaceRecognitionSystem
from db_operations import DatabaseManager

# Set page config
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state variables
if 'face_system' not in st.session_state:
    st.session_state.face_system = FaceRecognitionSystem()

if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()

def check_camera_availability():
    """Check if camera is available and return appropriate camera index"""
    # Try different camera indices
    for index in range(3):  # Try indices 0, 1, 2
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return index
    return None

# Main app
def main():
    st.title("Real-time Face Recognition System")
    
    # Check camera availability
    if 'camera_index' not in st.session_state:
        st.session_state.camera_index = check_camera_availability()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Register New Face", "View Attendance"])
    
    if page == "Home":
        home_page()
    elif page == "Register New Face":
        register_page()
    elif page == "View Attendance":
        attendance_page()

def home_page():
    st.header("Real-time Face Recognition")
    st.write("This system recognizes faces in real-time using your webcam.")
    
    # Start/stop webcam button
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False
    
    col1, col2 = st.columns(2)
    
    # Show camera status
    if st.session_state.camera_index is None:
        st.error("No webcam detected. Please connect a webcam and restart the application.")
        st.info("If you're using a virtual machine or WSL, make sure the webcam is properly shared with the environment.")
        
        # Troubleshooting info
        with st.expander("Troubleshooting Tips"):
            st.write("""
            ### Troubleshooting webcam issues:
            
            1. **Windows**: Check Device Manager to ensure your webcam is working properly.
            2. **Linux**: Try running `ls -l /dev/video*` in terminal to see available cameras.
            3. **macOS**: Check System Preferences > Security & Privacy > Camera permissions.
            4. **Docker/VM**: Ensure webcam is properly shared with the container/VM.
            
            If using WSL2, add these to your .wslconfig file:
            ```
            [wsl2]
            kernelCommandLine = "usbcore.usbfs_memory_mb=1024"
            ```
            
            Then restart WSL with `wsl --shutdown` and reopen.
            """)
        return
    
    with col1:
        if st.button("Start Webcam" if not st.session_state.webcam_running else "Stop Webcam"):
            st.session_state.webcam_running = not st.session_state.webcam_running
    
    with col2:
        confidence_threshold = st.slider("Recognition Confidence", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    
    # Display webcam feed with face recognition
    if st.session_state.webcam_running:
        stframe = st.empty()
        
        try:
            cap = cv2.VideoCapture(st.session_state.camera_index)
            if not cap.isOpened():
                st.error(f"Failed to open webcam at index {st.session_state.camera_index}")
                st.session_state.webcam_running = False
                return
            
            while st.session_state.webcam_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from webcam")
                    st.session_state.webcam_running = False
                    break
                
                # Process frame for face recognition
                face_locations, face_names, recognized_ids = st.session_state.face_system.recognize_faces(frame)
                
                # Draw face rectangles and names
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Draw rectangle around face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Draw label
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                
                # Display the frame
                stframe.image(frame, channels="BGR", use_column_width=True)
        
        except Exception as e:
            st.error(f"Error accessing webcam: {str(e)}")
            st.info("If you're running in a virtual environment or container, make sure the webcam is properly shared.")
        finally:
            # Release webcam when stopped
            if 'cap' in locals() and cap.isOpened():
                cap.release()

def register_page():
    st.header("Register New Face")
    st.write("Add a new person to the face recognition system.")
    
    # Form for registration
    with st.form("registration_form"):
        name = st.text_input("Person's Name")
        upload_method = st.radio("Choose upload method", ["Upload Image", "Capture from Webcam"])
        
        uploaded_file = None
        if upload_method == "Upload Image":
            uploaded_file = st.file_uploader("Upload an image with a clear face", type=["jpg", "jpeg", "png"])
        
        submit_button = st.form_submit_button("Register Face")
    
    # Handle webcam capture
    if upload_method == "Capture from Webcam":
        if st.session_state.camera_index is None:
            st.error("No webcam detected. Please use the 'Upload Image' option instead.")
        else:
            st.write("Click the button below to capture your face from webcam")
            if st.button("Capture Face"):
                with st.spinner("Opening webcam..."):
                    try:
                        cap = cv2.VideoCapture(st.session_state.camera_index)
                        if not cap.isOpened():
                            st.error(f"Could not open webcam at index {st.session_state.camera_index}")
                        else:
                            # Capture frame
                            ret, frame = cap.read()
                            if ret:
                                st.image(frame, channels="BGR", caption="Captured Image")
                                # Save the captured frame temporarily
                                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                                cv2.imwrite(temp_file.name, frame)
                                uploaded_file = temp_file.name
                            else:
                                st.error("Failed to capture image")
                    except Exception as e:
                        st.error(f"Error capturing image: {str(e)}")
                    finally:
                        if 'cap' in locals() and cap.isOpened():
                            cap.release()
    
    # Process registration
    if submit_button and name and uploaded_file:
        with st.spinner("Processing..."):
            try:
                # Read the image
                if isinstance(uploaded_file, str):  # From webcam
                    image = cv2.imread(uploaded_file)
                else:  # From file upload
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if image is None:
                    st.error("Failed to read image. Please try again.")
                    return
                
                # Register the face
                success, message = st.session_state.face_system.register_new_face(name, image)
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
                
                # Clean up temp file if from webcam
                if isinstance(uploaded_file, str) and os.path.exists(uploaded_file):
                    os.unlink(uploaded_file)
                    
            except Exception as e:
                st.error(f"Error during registration: {str(e)}")

def attendance_page():
    st.header("Attendance Records")
    
    # Get attendance records
    records = st.session_state.db_manager.get_attendance_records()
    
    if records:
        # Convert to DataFrame for better display
        import pandas as pd
        df = pd.DataFrame(records)
        st.dataframe(df)
        
        # Download option
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Attendance Data",
            data=csv,
            file_name="attendance_records.csv",
            mime="text/csv"
        )
    else:
        st.info("No attendance records found")

if __name__ == "__main__":
    main() 