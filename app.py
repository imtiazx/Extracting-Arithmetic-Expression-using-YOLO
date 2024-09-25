# Import required libraries
import streamlit as st
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# Load custom CSS
with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    st.title("**Extract and Calculate Arithmetic Expression**")

    # File uploader restricted to specific image types
    file_uploaded = st.file_uploader("Choose an image file", type=["png", "jpeg", "jpg", "webp"])
    st.write("Download sample images from this [dataset](https://github.com/imtiazx/Extracting-Arithmetic-Expression-using-YOLO/tree/main/dataset).")
    
    class_btn = st.button("Extract")  

    if file_uploaded is not None:
        # Check if the file is a valid image before proceeding
        try:
            image = Image.open(file_uploaded)
            image = np.array(image)
            st.write("Sample Image")
            st.image(image, use_column_width=True)  # Display the uploaded image
        except Exception as e:
            st.error("Error: Unable to process the file. Please upload a valid image file.")
            return

    if class_btn:
        if file_uploaded is None:
            st.write("Please upload an image before proceeding.")
        else:
            with st.spinner("Model is working..."):
                plt.imshow(image)
                plt.axis("off")
                time.sleep(1)
                st.success("See the result below.")
                st.write(extract(image))

def extract(img):
    yolov4_cfg = "model/large_yolov4_custom.cfg"
    trained_model = "model/yolov4-custom_last.weights"
    net = cv2.dnn.readNetFromDarknet(yolov4_cfg, trained_model)

    # Define classes for arithmetic symbols and digits
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '%']
    hight, width, _ = img.shape
    
    # Preprocess image for YOLO
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_name = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_name)

    # Initialize lists for detected boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Parse YOLO layer outputs
    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]            
            if confidence > 0.4:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3] * hight)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
            
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression to filter boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    # Initialize lists for arithmetic expression components
    L = []
    X = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y), font, 2, color, 2)
            L.append(label)
            X.append(x)

    # Sort the detected components by their x-coordinate to form the arithmetic expression
    zipped = zip(X, L)
    sorted_zipped = sorted(zipped)
    sorted_list = [element for _, element in sorted_zipped]
    
    # Form the expression and evaluate it
    Answer = ''.join(sorted_list)
    return f'The arithmetic expression is: {Answer}.\nThe value of the expression is: {str(round(eval(Answer), 2))}'

if __name__ == "__main__":
    main()
