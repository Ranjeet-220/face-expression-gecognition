import cv2
import argparse
import numpy as np
import tensorflow as tf
import os
from src.face_detector import FaceDetector
from src.model import build_model
from src.preprocessing import prepare_for_inference
from src.video_stream import VideoStream
from src.config import EMOTIONS, DEFAULT_MODEL_PATH

def load_emotion_model(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train.py or provide a valid path.")
        
        # Determine if we should build a dummy model for code verification
        # If the user just wants to verify the pipeline (as per plan), we can return an uninitialized model
        # BUT this will give garbage predictions.
        print("Building uninitialized model for demonstration/verification purposes...")
        return build_model()

    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def draw_results(frame, rects, model):
    for (x, y, w, h) in rects:
        # Extract face
        face_roi = frame[y:y+h, x:x+w]
        
        # Preprocess
        input_data = prepare_for_inference(face_roi)
        
        if input_data is not None:
            # Predict
            preds = model.predict(input_data, verbose=0)[0]
            label_idx = np.argmax(preds)
            label = EMOTIONS[label_idx]
            confidence = preds[label_idx]
            
            # Draw
            text = f"{label}: {confidence:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw confidence bar (optional visualization)
            bar_x = x
            bar_y = y + h + 10
            # for i, (emotion, prob) in enumerate(zip(EMOTIONS, preds)):
            #     text_prob = f"{emotion}: {prob:.2f}"
            #     cv2.putText(frame, text_prob, (bar_x, bar_y + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame

def run_webcam(model_path, source=0):
    detector = FaceDetector()
    model = load_emotion_model(model_path)
    if model is None: return

    print(f"Starting video stream from source: {source}...")
    vs = VideoStream(src=source)
    
    while True:
        frame = vs.read()
        if frame is None:
            break
            
        rects = detector.detect_faces(frame)
        frame = draw_results(frame, rects, model)
        
        cv2.imshow("Face Emotion Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    vs.stop()
    cv2.destroyAllWindows()

def run_image(image_path, model_path):
    detector = FaceDetector()
    model = load_emotion_model(model_path)
    if model is None: return

    if not os.path.exists(image_path):
        print(f"Error: Image not found - {image_path}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print("Failed to load image.")
        return

    rects = detector.detect_faces(frame)
    frame = draw_results(frame, rects, model)
    
    cv2.imshow("Output", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_image(image_path, model_path, save_path=None):
    detector = FaceDetector()
    model = load_emotion_model(model_path)
    if model is None: return

    if not os.path.exists(image_path):
        print(f"Error: Image not found - {image_path}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print("Failed to load image.")
        return

    rects = detector.detect_faces(frame)
    frame = draw_results(frame, rects, model)
    
    if save_path:
        cv2.imwrite(save_path, frame)
        print(f"Output saved to {save_path}")
    else:
        cv2.imshow("Output", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Face Emotion Recognition")
    parser.add_argument("--mode", type=str, choices=["webcam", "image"], default="webcam", help="Mode: webcam or image")
    parser.add_argument("--image_path", type=str, help="Path to image file (required for image mode)")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Path to trained model file")
    parser.add_argument("--save_output", type=str, help="Path to save output image (optional)")
    parser.add_argument("--source", type=str, default="0", help="Video source: '0' for webcam, or URL for IP camera")
    
    args = parser.parse_args()
    
    if args.mode == "image":
        if not args.image_path:
            print("Error: --image_path is required for image mode.")
        else:
            run_image(args.image_path, args.model_path, args.save_output)
    else:
        # Handle source argument
        source = args.source
        if source.isdigit():
            source = int(source)
        run_webcam(args.model_path, source=source)
