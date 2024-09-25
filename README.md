# Extracting-Arithmetic-Expression-using-YOLO

## Overview
This project was part of an in-house [hackathon on Kaggle](https://www.kaggle.com/competitions/computer-vision-1/overview) hosted by Bridgei2i, where it won first prize worth ₹50,000 INR. The challenge involved building a solution that utilizes computer vision techniques to analyze images containing arithmetic expressions and correctly identify each digit and symbol in the correct sequence. The solution was evaluated on the basis of Levenshtein distance for accuracy.

## Problem Statement
The goal was to create a solution that can:
- Analyze an image containing arithmetic expressions.
- Detect and classify each digit (0-9) and arithmetic symbol (e.g., +, -, *, /, %) in the correct sequence.
- Output the recognized arithmetic expression, which is then evaluated using Levenshtein distance for accuracy.

## Solution
The solution is simple and efficient, designed to work well with the given dataset. It employs a YOLOv4 custom model to detect digits and symbols in the images. The model's output is a sequence of characters that represent the arithmetic expression in the image. You can test the solution on this [Streamlit app](https://imtiazx-extracting-arithmetic-expression-using-yolo-app-370utz.streamlit.app/).

## Repository Structure
This repository contains the following files and folders:

- **dataset/**: Contains the training and testing images.
- **model/**: Holds the configuration (`large_yolov4_custom.cfg`) and weights (`yolov4-custom_last.weights`) files for the custom YOLOv4 model.
- **app.py**: The main application script that allows you to upload an image and get the arithmetic expression extracted from it.
- **custom.css**: Custom CSS file for styling the Streamlit web interface.
- **requirements.txt**: A list of necessary Python packages to install for running the project.

## Instructions to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/imtiazx/Extracting-Arithmetic-Expression-using-YOLO.git
2. Navigate to the repository folder:
   ```bash
   cd Extracting-Arithmetic-Expression-using-YOLO
3. pip install -r requirements.txt
   ```bash
   pip install -r requirements.txt
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
