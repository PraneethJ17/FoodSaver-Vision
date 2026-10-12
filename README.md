# 🍽️ FoodSaver Vision

### College Canteen Food Waste Estimator

## About the Project

FoodSaver Vision is a Computer Vision project that uses Python and OpenCV to analyse plate photographs and estimate the visible area of pixels that differ in colour from a selected clean-plate reference.

**Note:** This project estimates visible image area. It does not measure actual food weight, calories, or exact food waste.

## Features

* Upload JPG, JPEG, and PNG images.
* Select the plate region using an adjustable ellipse.
* Select a clean-plate reference patch.
* Detect colour differences using OpenCV.
* Reduce noise using morphological operations.
* Display the original image, binary detection mask, and red-highlighted overlay.
* Calculate the estimated visible area percentage.
* Download the overlay image.

## Technologies Used

* Python
* OpenCV
* NumPy
* Streamlit
* Pillow

## Installation and Setup

Install the required libraries:

```bash
python -m pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

Open your browser at:

http://localhost:8501

## How to Use

1. Upload a photograph of a plate.
2. Adjust the green ellipse to fit the plate.
3. Move the blue square onto a clean area of the plate.
4. Adjust the colour-distance threshold.
5. Review the detection mask, overlay, and estimated visible area percentage.
6. Download the overlay image.

## Methodology

1. Load and validate the image.
2. Convert RGB image data to CIELAB colour space.
3. Calculate the median colour of the reference patch.
4. Calculate pixel colour distances.
5. Apply thresholding and restrict detection to the selected plate region.
6. Apply morphological operations to reduce noise.
7. Calculate the estimated visible area percentage.

## Limitations

* Shadows and reflections may affect results.
* Patterned plates can cause false detections.
* Food similar in colour to the plate may be missed.
* Results depend on image quality and manual selections.
* The project uses classical image processing, not a trained AI model.
* The percentage is not a measurement of actual food weight or exact food waste.

## Future Enhancements

* Automatic plate detection.
* Food segmentation using a trained model.
* Evaluation using a documented image dataset.
* Improved estimation methods using additional measurements.

## Project Information

**Project:** FoodSaver Vision
**Track:** Computer Vision
**Event:** HackZen 2.0
