from tflite_runtime.interpreter import Interpreter
import numpy as np
from PIL import Image
import time
import cv2

model = "/home/gemma/Arinapin/Speech recognition/gestos2/clients_Fd6ROLFygUROdCkBbME0KJNmUAG3_L29Fg0jIi5bMpRMy9fwX_lD8eTCmzjR40dXFcbock_lD8eTCmzjR40dXFcbock.tflite"
labels_path = "/home/gemma/Arinapin/Speech recognition/gestos2/labels.txt"

interpreter = Interpreter(model_path=model)
interpreter.allocate_tensors()

# Load labels from file
def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

labels = load_labels(labels_path)

# Function to perform inference on a frame
def perform_inference_stream(frame):
    _, height, width, _ = interpreter.get_input_details()[0]['shape']

    # Resize the frame
    img_resized = cv2.resize(frame, (width, height))

    # Preprocess the frame
    input_data = np.array(img_resized, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)

    # Set input tensor
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    tensor_result = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])[0]
    results = np.squeeze(tensor_result)

    # Get top 5 results
    top_k = results.argsort()[-5:][::-1]

    result = ''
    for i in top_k:
        result += ('{:08.6f}: {}'.format(float(tensor_result[i]), labels[i])) + "\n"

    return result

# Function to capture video frames and perform inference
def capture_frames_and_inference():
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Perform inference on the frame
            result = perform_inference_stream(frame)

            # Display the result
            cv2.putText(frame, result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Video Inference', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Start capturing frames and performing inference
capture_frames_and_inference()
