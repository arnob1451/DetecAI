<<<<<<< HEAD
=======
import warnings
import time
from PIL import Image
from ultralytics import YOLO
>>>>>>> origin
import pandas as pd
from pathlib import Path
from config import *
from utils import logger, log_execution_time, validate_image, cleanup_cache, setup_logging
# Set page configuration as the first Streamlit command
st.set_page_config(page_title="DectecAI-Smart Object & Edge Detection App", layout="wide")

=======
# Setup logging
setup_logging()
>>>>>>> origin

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="DetecAI-Smart Object & Edge Detection App", layout="wide")


@log_execution_time
def apply_edge_detection(method: str, image: np.ndarray, params: dict) -> np.ndarray:
    """Apply selected edge detection algorithm"""
    try:
        if method == "Canny":
            return cv2.Canny(image, params['threshold1'], params['threshold2'])
        elif method == "Sobel":
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=params['kernel_size'])
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=params['kernel_size'])
            edges = cv2.magnitude(sobelx, sobely)
        elif method == "Laplacian":
            edges = cv2.Laplacian(image, cv2.CV_64F, ksize=params['laplacian_kernel'])
            
        return cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    except Exception as e:
        logger.error(f"Edge detection failed: {str(e)}")
        raise RuntimeError(f"Edge detection failed: {str(e)}")

@log_execution_time
def detect_objects(model, image: np.ndarray, selected_classes: list) -> tuple:
    """Run YOLO object detection and return results with visualization"""
    try:
        start_time = time.time()
        results = model(image)
        detections = []
        counts = {}
        viz_image = image.copy()
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                
                if selected_classes and class_name not in selected_classes:
                    continue
                    
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(viz_image, f"{class_name} {conf:.2f}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                detections.append({
                    "class": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })
                counts[class_name] = counts.get(class_name, 0) + 1
                
        return viz_image, detections, counts, time.time()-start_time
    except Exception as e:
        logger.error(f"Object detection failed: {str(e)}")
        raise RuntimeError(f"Object detection failed: {str(e)}")

@log_execution_time
def generate_heatmap(edges: np.ndarray) -> np.ndarray:
    """Generate edge intensity heatmap"""
    try:
        return cv2.applyColorMap(edges, cv2.COLORMAP_JET)
    except Exception as e:
        logger.error(f"Heatmap generation failed: {str(e)}")
        raise RuntimeError(f"Heatmap generation failed: {str(e)}")

def apply_theme(high_contrast):
    """Apply theme based on contrast mode"""
    if high_contrast:
        st.markdown("""
            <style>
            .stImage > img {max-width: 100%; height: auto; border: 2px solid #000000; border-radius: 5px;}
            .stSidebar {background-color: #FFFFFF; border-right: 2px solid #000000;}
            .stExpander {background-color: #FFFFFF; border: 2px solid #000000; border-radius: 5px;}
            h1, h2, h3, p, div, label, span {color: #000000 !important;}
            .stButton>button {background-color: #000000; color: #FFFFFF; border-radius: 5px; border: 2px solid #000000;}
            .stButton>button:hover {background-color: #333333;}
            .stDataFrame {background-color: #FFFFFF; border: 2px solid #000000; border-radius: 5px;}
            .main {background-color: #FFFFFF; padding: 20px;}
            </style>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .main {background-color: #000000; padding: 20px;}
            .stImage > img {max-width: 100%; height: auto; border: 1px solid #ffffff; border-radius: 5px;}
            .stSidebar {background-color: #000000;}
            .stExpander {background-color: #000000; border: 1px solid #ffffff; border-radius: 5px;}
            h1, h2, h3, p, div, label, span {color: #ffffff !important;}
            .stButton>button {background-color: #333333; color: #ffffff; border-radius: 5px; border: none;}
            .stButton>button:hover {background-color: #555555;}
            .stDataFrame {background-color: #000000; border: 1px solid #ffffff; border-radius: 5px;}
            </style>
            """, unsafe_allow_html=True)

def main():
    try:
        st.title("DetecAI-Smart Object and Edge Detection Application")
        
        # High Contrast Mode Toggle
        high_contrast = st.sidebar.checkbox("High Contrast Mode", value=False)
        apply_theme(high_contrast)
        
        # Load model
        try:
            model = load_yolo_model()
        except Exception as e:
            st.error(str(e))
            st.stop()

        # Input Selection
        input_method = st.sidebar.radio("Input Method", ["File Upload", "Webcam Capture"])
        
        # File upload/webcam capture section
        with st.container():
            st.subheader("Image Input")
            img = None
            
<<<<<<< HEAD
            if edge_method == "Canny":
                edge_params['threshold1'] = st.slider("Canny Threshold 1", 0, 500, 100, key="canny_t1")
                edge_params['threshold2'] = st.slider("Canny Threshold 2", 0, 500, 200, key="canny_t2")
            elif edge_method == "Sobel":
                edge_params['kernel_size'] = st.slider("Kernel Size", 3, 15, 3, step=2, key="kernel_size")
            elif edge_method == "Laplacian":
                edge_params['laplacian_kernel'] = st.slider("Laplacian Kernel", 3, 15, 3, step=2, key="laplacian_kernel")


        # Preprocessing options
        with st.expander("Preprocessing Options", expanded=True):
            preprocess_params = {
                'gaussian': st.checkbox("Gaussian Blur", True, key="gaussian"),
                'gaussian_kernel': st.slider("Gaussian Kernel", 3, 15, 5, step=2, key="gaussian_kernel"),
                'threshold': st.checkbox("Adaptive Thresholding", key="threshold"),
                'morph_kernel': st.slider("Morph Kernel", 3, 15, 3, step=2, key="morph_kernel")
            }

        # Object detection options
        with st.expander("Object Detection Settings", expanded=True):
            yolo_enabled = st.checkbox("Enable YOLO Detection", True, key="yolo_enabled")
            draw_contours = st.checkbox("Draw Edge Contours", True, key="draw_contours")
            class_options = list(model.names.values()) if yolo_enabled else []
            selected_classes = st.multiselect("Filter Classes", 
                                             class_options, 
                                             default=class_options[:DEFAULT_CLASSES_TO_SHOW] if class_options else [],
                                             key="class_filter")

    # ======================
    # IMAGE PROCESSING
    # ======================
    try:
        # Apply preprocessing
        start_preprocess = time.time()
        processed_img = apply_preprocessing(img_gray, preprocess_params)
        preprocess_time = time.time() - start_preprocess

        # Edge detection
        start_edge = time.time()
        edges = apply_edge_detection(edge_method, processed_img, edge_params)
        edge_time = time.time() - start_edge
        
        contour_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        if draw_contours:
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

        # Object detection
        yolo_image = img_rgb.copy()
        detections = []
        counts = {}
        yolo_time = 0
        if yolo_enabled:
            yolo_image, detections, counts, yolo_time = detect_objects(model, img_rgb, selected_classes)

        # Generate heatmap
        start_heatmap = time.time()
        heatmap = generate_heatmap(edges)
        heatmap_time = time.time() - start_heatmap

    except Exception as e:
        st.error(f"Processing error: {str(e)}")
        st.stop()


    with st.container():
        st.subheader("Processing Results")
        
        # Performance Metrics
        with st.expander("Performance Metrics", expanded=True):
            metrics = {
                "Preprocessing": f"{preprocess_time*1000:.2f} ms",
                "Edge Detection": f"{edge_time*1000:.2f} ms",
                "Object Detection": f"{yolo_time*1000:.2f} ms" if yolo_enabled else "N/A",
                "Heatmap Generation": f"{heatmap_time*1000:.2f} ms"
            }
            st.table(pd.DataFrame(list(metrics.items()), columns=["Process", "Time"]))

        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(adjusted_img, caption="Adjusted Input Image", use_container_width=True)
            st.image(Image.fromarray(processed_img if len(processed_img.shape) == 2 else cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)), 
                     caption="Preprocessed Image", use_container_width=True)
        
        with col2:
            st.image(Image.fromarray(contour_img), caption=f"{edge_method} Edges with Contours", use_container_width=True)
            if yolo_enabled:
                st.image(Image.fromarray(yolo_image), caption="YOLO Object Detection", use_container_width=True)
            st.image(Image.fromarray(heatmap), caption="Edge Intensity Heatmap", use_container_width=True)

        # Download options
        with st.expander("Download Options", expanded=False):
            buf = io.BytesIO()
            if yolo_enabled:
                Image.fromarray(yolo_image).save(buf, format="PNG")
=======
            if input_method == "File Upload":
                uploaded_file = st.file_uploader("Choose an image...", type=SUPPORTED_FORMATS)
                if uploaded_file:
                    valid, message = validate_image(uploaded_file)
                    if valid:
                        img = Image.open(uploaded_file)
                    else:
                        st.error(message)
                        st.stop()
>>>>>>> origin
            else:
                webcam_img = st.camera_input("Take a picture...")
                if webcam_img:
                    img = Image.open(webcam_img)

            if not img:
                st.info("Upload an image or use webcam to begin processing.")
                return

            # Image adjustments
            with st.sidebar.expander("Image Adjustments", expanded=True):
                brightness = st.slider("Brightness", 0, 200, 100)
                contrast = st.slider("Contrast", 0, 200, 100)
                saturation = st.slider("Saturation", 0, 200, 100)
                
            adjusted_img = adjust_image_properties(img, brightness, contrast, saturation)
            img_gray, img_rgb = preprocess_image(adjusted_img)

            # Processing parameters
            with st.sidebar:
                st.header("Processing Parameters")
                
                with st.expander("Edge Detection Settings", expanded=True):
                    edge_method = st.selectbox("Edge Detection Method", EDGE_METHODS)
                    edge_params = {}
                    
                    if edge_method == "Canny":
                        edge_params['threshold1'] = st.slider("Threshold 1", 0, 500, 100)
                        edge_params['threshold2'] = st.slider("Threshold 2", 0, 500, 200)
                    elif edge_method == "Sobel":
                        edge_params['kernel_size'] = st.slider("Kernel Size", 3, 15, 3, step=2)
                    elif edge_method == "Laplacian":
                        edge_params['laplacian_kernel'] = st.slider("Kernel Size", 3, 15, 3, step=2)

                with st.expander("Preprocessing Options", expanded=True):
                    preprocess_params = {
                        'gaussian': st.checkbox("Gaussian Blur", True),
                        'gaussian_kernel': st.slider("Gaussian Kernel", 3, 15, 5, step=2),
                        'threshold': st.checkbox("Adaptive Thresholding"),
                        'hist_eq': st.checkbox("Histogram Equalization"),
                        'morph': st.checkbox("Morphological Operations"),
                        'morph_kernel': st.slider("Morph Kernel", 3, 15, 3, step=2)
                    }

                with st.expander("Object Detection Settings", expanded=True):
                    yolo_enabled = st.checkbox("Enable YOLO Detection", True)
                    draw_contours = st.checkbox("Draw Edge Contours", True)
                    if yolo_enabled:
                        class_options = list(model.names.values())
                        selected_classes = st.multiselect(
                            "Filter Classes",
                            class_options,
                            default=class_options[:DEFAULT_CLASSES_TO_SHOW]
                        )

            # Process images
            processed_img = apply_preprocessing(img_gray, preprocess_params)
            edges = apply_edge_detection(edge_method, processed_img, edge_params)
            
            contour_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            if draw_contours:
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

            # Object detection
            detections = []
            counts = {}
            yolo_time = 0
            if yolo_enabled:
                yolo_image, detections, counts, yolo_time = detect_objects(model, img_rgb, selected_classes)

            # Generate heatmap
            heatmap = generate_heatmap(edges)

            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(adjusted_img, caption="Input Image", use_column_width=True)
                st.image(processed_img, caption="Preprocessed Image", use_column_width=True)
            
            with col2:
                st.image(contour_img, caption=f"{edge_method} Edges", use_column_width=True)
                if yolo_enabled:
                    st.image(yolo_image, caption="Object Detection", use_column_width=True)
                st.image(heatmap, caption="Edge Heatmap", use_column_width=True)

            # Detection results
            if yolo_enabled and detections:
                st.subheader("Detection Results")
                st.dataframe(pd.DataFrame(detections))
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    buf = io.BytesIO()
                    Image.fromarray(yolo_image).save(buf, format="PNG")
                    st.download_button(
                        "Download Processed Image",
                        buf.getvalue(),
                        "processed_image.png",
                        "image/png"
                    )
                
                with col2:
                    st.download_button(
                        "Download Detection Results",
                        json.dumps(detections, indent=2),
                        "detections.json",
                        "application/json"
                    )

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()