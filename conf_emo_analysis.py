import cv2
import numpy as np
import os
import argparse
from openvino import Core
from pathlib import Path

# Labels for emotions
EMOTIONS = ['neutral', 'happy', 'sad', 'surprise', 'anger']

def initialize_models(device='NPU'):
    """
    Initialize OpenVINO models for face detection and emotion recognition.
    
    Args:
        device (str): Target device for inference ('CPU', 'GPU', 'NPU')
    
    Returns:
        tuple: (face_detector, emotion_classifier, core)
    """
    core = Core()
    
    # Initialize face detection model (face-detection-retail-0004)
    face_model_xml = "C:\\Users\\test\\source\\repos\\OpenVINO-Samples\\intel\\face-detection-retail-0004\\FP16-INT8\\face-detection-retail-0004.xml"
    face_detector = core.compile_model(face_model_xml, device)
    
    # Initialize emotion recognition model (emotions-recognition-retail-0003)
    emotion_model_xml = "C:\\Users\\test\\source\\repos\\qwen2.5-vl\\models\\emotions-recognition-retail-0003\\FP16\\emotions-recognition-retail-0003.xml"
    emotion_classifier = core.compile_model(emotion_model_xml, device)
    
    return face_detector, emotion_classifier, core

def preprocess_frame(frame):
    """Preprocess frame for model input"""
    resized = cv2.resize(frame, (300, 300))
    input_tensor = np.expand_dims(resized.transpose(2, 0, 1), 0)
    return input_tensor.astype(np.float32)

def analyze_video_emotion(video_path, sample_rate=1, device='NPU'):
    """
    Analyze a video clip for person detection and emotion recognition.
    
    Args:
        video_path (str): Path to the MP4 video file
        sample_rate (int): Process every nth frame (default=1)
        device (str): Target device for inference ('CPU', 'GPU', 'NPU')
    
    Returns:
        dict: Analysis results containing:
            - has_person (bool): Whether any person was detected
            - emotions (list): List of detected emotions with timestamps
            - dominant_emotion (str): Most frequent emotion detected
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
      
    # Initialize OpenVINO models
    face_detector, emotion_classifier, core = initialize_models(device)
    face_output = face_detector.output(0)
    emotion_output = emotion_classifier.output(0)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    results = {
        'has_person': False,
        'emotions': [],
        'dominant_emotion': None
    }
    
    frame_count = 0
    emotion_counts = {}
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every nth frame based on sample_rate
            if frame_count % sample_rate != 0:
                frame_count += 1
                continue
                  
            # Preprocess frame for face detection
            input_tensor = preprocess_frame(frame)
            
            # Run face detection
            faces = face_detector([input_tensor])[face_output]
            faces = faces.reshape(-1, 7)
            
            # Filter valid detections (confidence > 0.5)
            valid_detections = faces[faces[:, 2] > 0.5]
            
            if len(valid_detections) > 0:
                results['has_person'] = True
                
                # Process each detected face
                for detection in valid_detections:
                    # Extract face coordinates
                    xmin, ymin, xmax, ymax = detection[3:7] * np.array([frame.shape[1], frame.shape[0]] * 2)
                    xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
                    
                    # Extract and preprocess face for emotion recognition
                    face_roi = frame[ymin:ymax, xmin:xmax]
                    if face_roi.size == 0:
                        continue
                        
                    face_roi = cv2.resize(face_roi, (64, 64))  # Required size for emotion model
                    face_input = np.expand_dims(face_roi.transpose(2, 0, 1), 0).astype(np.float32)
                    
                    # Run emotion recognition
                    emotion_probs = emotion_classifier([face_input])[emotion_output]
                    emotion_probs = emotion_probs.reshape(-1)
                    dominant_emotion_idx = np.argmax(emotion_probs)
                    dominant_emotion = (EMOTIONS[dominant_emotion_idx], float(emotion_probs[dominant_emotion_idx]))
                    
                    timestamp = frame_count / fps
                    
                    # Store emotion with timestamp
                    results['emotions'].append({
                        'timestamp': timestamp,
                        'emotion': dominant_emotion[0],
                        'confidence': dominant_emotion[1]
                    })
                    
                    # Update emotion counts for overall dominant emotion
                    emotion_counts[dominant_emotion[0]] = emotion_counts.get(dominant_emotion[0], 0) + 1
            
            frame_count += 1
            
            # Optional: print progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Processing: {progress:.1f}% complete")
    
    finally:
        cap.release()
    
    # Determine the dominant emotion overall
    if emotion_counts:
        results['dominant_emotion'] = max(emotion_counts.items(), key=lambda x: x[1])[0]
    
    return results

def display_results(results):
    """
    Display the analysis results in a human-readable format.
    
    Args:
        results (dict): The results from analyze_video_emotion
    """
    print("\nVideo Analysis Results:")
    print("-" * 50)
    print(f"Person Detected: {'Yes' if results['has_person'] else 'No'}")
    
    if results['has_person']:
        print(f"\nDominant Emotion: {results['dominant_emotion']}")
        print("\nEmotion Timeline:")
        for entry in results['emotions']:
            print(f"Time: {entry['timestamp']:.2f}s - {entry['emotion']} ({entry['confidence']:.2%})")
    
    print("-" * 50)

def get_available_cameras():
    """
    Get a list of available camera devices.
    
    Returns:
        list: List of available camera indices.
    """
    available_cameras = []
    for i in range(5):  # Check first 5 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def analyze_from_camera(camera_idx=0, sample_rate=1, device='NPU'):
    """
    Analyze emotion from live camera feed without display.
    
    Args:
        camera_idx (int): Camera device index
        sample_rate (int): Process every nth frame
        device (str): Target device for inference
    """
    # Initialize models
    face_detector, emotion_classifier, core = initialize_models(device)
    face_output = face_detector.output(0)
    emotion_output = emotion_classifier.output(0)
    
    # Open camera
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise ValueError(f"Could not open camera with index {camera_idx}")
    
    print("Starting camera analysis... Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every nth frame
            if frame_count % sample_rate != 0:
                frame_count += 1
                continue
            
            # Detect faces and emotions
            input_tensor = preprocess_frame(frame)
            faces = face_detector([input_tensor])[face_output]
            faces = faces.reshape(-1, 7)
            valid_detections = faces[faces[:, 2] > 0.5]
            
            if len(valid_detections) > 0:
                print(f"\nFrame {frame_count}:")
                
            for detection in valid_detections:
                # Extract face ROI
                xmin, ymin, xmax, ymax = detection[3:7] * np.array([frame.shape[1], frame.shape[0]] * 2)
                xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
                
                face_roi = frame[ymin:ymax, xmin:xmax]
                if face_roi.size == 0:
                    continue
                    
                # Get emotion
                face_roi = cv2.resize(face_roi, (64, 64))
                face_input = np.expand_dims(face_roi.transpose(2, 0, 1), 0).astype(np.float32)
                
                emotion_probs = emotion_classifier([face_input])[emotion_output]
                emotion_probs = emotion_probs.reshape(-1)
                dominant_emotion_idx = np.argmax(emotion_probs)
                emotion = EMOTIONS[dominant_emotion_idx]
                confidence = float(emotion_probs[dominant_emotion_idx])
                
                # Print results
                print(f"  Person detected - Emotion: {emotion} (Confidence: {confidence:.2%})")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nAnalysis stopped by user")
    finally:
        cap.release()

def main():
    parser = argparse.ArgumentParser(description='Emotion Analysis from Video or Camera')
    parser.add_argument('--source', type=str, choices=['video', 'camera'], required=True,
                      help='Source type: video file or camera')
    parser.add_argument('--input', type=str,
                      help='Path to video file (required if source is video)')
    parser.add_argument('--camera-id', type=int, default=0,
                      help='Camera device index (default: 0)')
    parser.add_argument('--device', type=str, choices=['CPU', 'GPU', 'NPU'], default='NPU',
                      help='Device for inference (default: NPU)')
    parser.add_argument('--sample-rate', type=int, default=1,
                      help='Process every nth frame (default: 1)')
    
    args = parser.parse_args()
    
    try:
        if args.source == 'video':
            if not args.input:
                parser.error("--input is required when source is 'video'")
            results = analyze_video_emotion(args.input, args.sample_rate, args.device)
            display_results(results)
        else:  # camera
            available_cameras = get_available_cameras()
            if not available_cameras:
                print("No cameras found!")
                return
            if args.camera_id not in available_cameras:
                print(f"Camera {args.camera_id} not available. Available cameras: {available_cameras}")
                return
            analyze_from_camera(args.camera_id, args.sample_rate, args.device)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()