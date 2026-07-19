import cv2
import mediapipe as mp

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh

# Set up the FaceMesh model
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Start video capture from the default camera
cap = cv2.VideoCapture(0)

print("Starting camera feed. Press 'q' to quit.")

while cap.isOpened():
    # Read a frame from the camera
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a natural, selfie-view display
    frame = cv2.flip(frame, 1)


    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # To improve performance, mark the image as not writeable
    # to pass by reference to MediaPipe
    
    # --- FIXED LINE ---
    rgb_frame.flags.writeable = False
    
    # Process the image and find face landmarks
    results = face_mesh.process(rgb_frame)

    # Mark the image as writeable again to draw on it
    
    # --- FIXED LINE ---
    rgb_frame.flags.writeable = True

    # --- Face Detection Logic ---
    face_present = False
    
    if results.multi_face_landmarks:
        face_present = True

    # --- Display Text ---
    if face_present:
        text = "Focused"
        color = (0, 255, 0)  # Green
    else:
        text = "Not Focused"
        color = (0, 0, 255)  # Red

    # Put the text on the frame
    cv2.putText(
        frame,
        text,
        (30, 50),  # Position (x, y) from top-left
        cv2.FONT_HERSHEY_SIMPLEX,  # Font type
        1,  # Font scale
        color,  # Font color
        2,  # Thickness
        cv2.LINE_AA  # Anti-aliasing
    )

    # Display the resulting frame
    cv2.imshow('MediaPipe FaceMesh Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Clean up
face_mesh.close()
cap.release()
cv2.destroyAllWindows()
print("Camera feed stopped.")