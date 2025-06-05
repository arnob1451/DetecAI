# DetecAI - Smart Object & Edge Detection App

An advanced computer vision application that combines object detection and edge detection capabilities using YOLOv8 and OpenCV.

## Features

- Object detection using YOLOv8
- Multiple edge detection methods (Canny, Sobel, Laplacian)
- Real-time webcam processing
- Image preprocessing options
- Performance analytics
- Edge intensity heatmaps
- High contrast mode
- Download processed results

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
streamlit run app.py
```

## Configuration

The application can be configured through the config.py file:

- `MODEL_PATH`: Path to YOLOv8 model
- `MAX_IMAGE_SIZE`: Maximum image dimension
- `LOG_LEVEL`: Logging level (INFO/DEBUG/ERROR)

## Production Deployment

1. Configure logging
2. Ensure cache directory exists and is writable
3. Use production-ready WSGI server

## License

MIT License

## Authors

Team Fantastic Four