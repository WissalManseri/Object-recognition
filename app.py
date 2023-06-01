import cv2
import tensorflow as tf
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Define the class labels
class_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Create a function to detect objects in a video stream
def detect_objects():
    # Open a video stream
    cap = cv2.VideoCapture(0)
    
    # Loop over the frames in the video stream
    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()
        
        # Resize the frame to the size required by the model
        resized_frame = cv2.resize(frame, (224, 224))
        
        # Preprocess the frame
        preprocessed_frame = tf.keras.applications.mobilenet_v2.preprocess_input(resized_frame)
        
        # Make a prediction on the frame
        predictions = model.predict(np.array([preprocessed_frame]))
        
        # Get the index of the most likely class
        class_index = np.argmax(predictions)
        
        # Get the label for the class
        class_label = class_labels[class_index]
        
        # Draw the label on the frame
        cv2.putText(frame, class_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Object detection', frame)
        
        # Wait for the user to press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video stream and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Call the detect_objects function to start detecting objects in the video stream
detect_objects()
