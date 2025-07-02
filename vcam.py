import cv2
import pyvirtualcam
import numpy as np
import argparse
import sys
import time
from openvino import Core

class FaceDetector:
    def __init__(self, model_path, device="GPU", confidence_threshold=0.3):
        ie = Core()
        
        available_devices = ie.available_devices
        if device not in available_devices:
            print(f"Device {device} not found. Available devices: {available_devices}")
            self.device = "CPU"
        else:
            self.device = device
            
        model = ie.read_model(model_path)
        try:
            self.compiled_model = ie.compile_model(model, self.device)
        except Exception as e:
            print(f"Error compiling model on {self.device}: {e}")
            print("Falling back to CPU...")
            self.device = "CPU"
            self.compiled_model = ie.compile_model(model, self.device)
            
        self.output_layer = self.compiled_model.output(0)
        self.confidence_threshold = confidence_threshold

    def detect_faces(self, frame):
        resized_frame = cv2.resize(frame, (300, 300))
        input_data = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)
        results = self.compiled_model([input_data])[self.output_layer]

        faces = []
        for detection in results[0][0]:
            confidence = detection[2]
            if confidence > self.confidence_threshold:
                xmin = int(detection[3] * frame.shape[1])
                ymin = int(detection[4] * frame.shape[0])
                xmax = int(detection[5] * frame.shape[1])
                ymax = int(detection[6] * frame.shape[0])
                faces.append((xmin, ymin, xmax, ymax))
        return faces
    
class PersonDetector:
    def __init__(self, model_path, device="NPU", confidence_threshold=0.6):
        ie = Core()
        
        available_devices = ie.available_devices
        if device not in available_devices:
            print(f"Device {device} not found. Available devices: {available_devices}")
            self.device = "CPU"
        else:
            self.device = device
            
        model = ie.read_model(model_path)
        try:
            self.compiled_model = ie.compile_model(model, self.device)
        except Exception as e:
            print(f"Error compiling model on {self.device}: {e}")
            print("Falling back to CPU...")
            self.device = "CPU"
            self.compiled_model = ie.compile_model(model, self.device)
            
        self.output_layer = self.compiled_model.output(0)
        self.confidence_threshold = confidence_threshold

    def detect_persons(self, frame):
        resized_frame = cv2.resize(frame, (1344, 800))
        input_data = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)
        results = self.compiled_model([input_data])[self.output_layer]
        print(results.shape)

        persons = []
        if results.shape[0] > 0 and results.shape[1] > 0:
            for detection in results[0][0]:
                confidence = detection[2]
                print(f"Detection : {detection}")
                if confidence > self.confidence_threshold:
                    xmin = int(detection[3] * frame.shape[1])
                    ymin = int(detection[4] * frame.shape[0])
                    xmax = int(detection[5] * frame.shape[1])
                    ymax = int(detection[6] * frame.shape[0])
                    persons.append((xmin, ymin, xmax, ymax))
        return persons
    
def crop_face_with_padding(frame, face_rect, target_size=(720, 540)):
    # Check if the frame is a valid numpy array
    if not isinstance(frame, np.ndarray):
        return False, np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # Check if the face_rect is a tuple of 4 integers
    if not isinstance(face_rect, tuple) or len(face_rect) != 4:
        return False, np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    
    if not all(isinstance(coord, int) for coord in face_rect):        
        return False, np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # Check if the target_size is a tuple of 2 integers
    if not isinstance(target_size, tuple) or len(target_size) != 2:
        return False, np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    if not all(isinstance(dim, int) for dim in target_size):
        return False, np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    xmin, ymin, xmax, ymax = face_rect
    
    face_width = xmax - xmin
    face_height = ymax - ymin

    # Calculate the center of the face
    face_center_x = (xmin + xmax) // 2
    face_center_y = (ymin + ymax) // 2

    # Calculate crop boundaries with 1.5x zoom
    crop_width = face_width * 1.5
    crop_height = face_height * 1.5

    crop_xmin = face_center_x - crop_width // 2
    crop_xmax = face_center_x + crop_width // 2
    crop_ymin = face_center_y - crop_height // 2
    crop_ymax = face_center_y + crop_height // 2

    # Check if crop boundaries are within the frame
    if crop_xmin < 0 or crop_ymin < 0 or crop_xmax > frame.shape[1] or crop_ymax > frame.shape[0]:
        #print("Face rect is out of boundary")
        return False, np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    # Calculate the ratio of the crop and target size
    crop_ratio = crop_width / crop_height
    target_ratio = target_size[0] / target_size[1]

    # Determine the maximum scale factor that the crop can be enlarged without exceeding the target size
    max_scale_factor_width = target_size[0] / crop_width
    max_scale_factor_height = target_size[1] / crop_height
    max_scale_factor = min(max_scale_factor_width, max_scale_factor_height)

    # Calculate the new crop dimensions based on the maximum scale factor
    new_crop_width = int(crop_width * max_scale_factor)
    new_crop_height = int(crop_height * max_scale_factor)

    # Calculate the padding values
    delta_w = target_size[0] - new_crop_width
    delta_h = target_size[1] - new_crop_height
    pad_top, pad_bottom = delta_h // 2, delta_h - (delta_h // 2)
    pad_left, pad_right = delta_w // 2, delta_w - (delta_w // 2)

    # Crop the frame
    cropped = frame[int(crop_ymin):int(crop_ymax), int(crop_xmin):int(crop_xmax)]

    # Resize the cropped frame
    resized = cv2.resize(cropped, (new_crop_width, new_crop_height))

    # Add padding to the left side of the resized frame
    color = [0, 0, 0]  # Black color
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=color)
    return True, padded


def create_virtual_camera(detector, emotion_classifier, video_source=0, target_fps=30, mirror_mode=False, count_mode = False):
    # Open the video source (webcam or video file)
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open the video source {video_source}")
        sys.exit(1)

    # Set virtual camera resolution and FPS
    frame_width = 1280
    frame_height = 720
    fps = target_fps
    frame_interval = 1.0 / fps  # Time per frame in seconds
    
    # Get source FPS for video files to calculate playback speed
    if isinstance(video_source, str):
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0:
            source_fps = target_fps
        playback_speed = source_fps / target_fps
        print(f"Source FPS: {source_fps:.2f}, Playback Speed Adjustment: {playback_speed:.2f}x")

    # Configure video capture settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Get total frames for video files
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if isinstance(video_source, str) else -1
    
    # Face detection interval (only detect faces every N frames)
    detection_interval = 30  # Increased frequency for better tracking
    frame_count = 0
    last_faces = []  # Store last detected faces
    last_cropped_frame = None  # Store last cropped frame
    
    # FPS control variables
    frame_times = []
    fps_update_interval = 30  # Update FPS display every 30 frames
    
    try:
        with pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=fps) as cam:
            print(f'Virtual camera created: {cam.device} (Target FPS: {fps})')
            last_frame_time = time.time()
            
            while True:
                frame_start_time = time.time()
                  # Calculate time to wait to maintain target FPS
                elapsed = frame_start_time - last_frame_time
                target_elapsed = frame_interval
                
                # For video files, adjust frame reading based on source FPS
                if isinstance(video_source, str):
                    frames_to_skip = int(playback_speed - 1)
                    if frames_to_skip > 0:
                        for _ in range(frames_to_skip):
                            cap.read()  # Skip frames to maintain target FPS
                    elif elapsed < target_elapsed:
                        # If video is slower than target FPS, wait for next frame
                        time.sleep(target_elapsed - elapsed)
                else:
                    # For webcam, simple FPS control
                    if elapsed < target_elapsed:
                        time.sleep(target_elapsed - elapsed)
                
                ret, frame = cap.read()
                
                # For video files, loop back to the beginning when reaching the end
                if not ret and isinstance(video_source, str):
                    print("Reached end of video file, restarting...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                elif not ret:
                    print("Error: Could not read frame from the video source")
                    break
                
                output_frame = frame

                # Determine if the detector is a FaceDetector or PersonDetector
                if isinstance(detector, FaceDetector):
                    # Only detect faces every detection_interval frames
                    if frame_count % detection_interval == 0:
                        faces = detector.detect_faces(frame)
                        last_faces = faces
                    else:
                        faces = last_faces

                    if faces:
                        if count_mode:
                            #if len(faces) != len(last_faces):
                            cv2.putText(frame, f"Detected {len(faces)} faces", (10, frame_height - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            for xmin, ymin, xmax, ymax in faces:
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        else:
                            # Crop the first detected face with padding
                            face_rect = faces[0]
                            ret, cropped_frame = crop_face_with_padding(frame, face_rect, (frame_width, frame_height))
                            if ret:
                                if emotion_classifier is not None:
                                    # Preprocess the cropped frame for emotion recognition
                                    input_img = cv2.resize(cropped_frame.copy(), (64, 64))
                                    input_img = input_img.transpose((2, 0, 1))
                                    input_img = np.expand_dims(input_img, axis=0)

                                    # Perform emotion recognition
                                    results = emotion_classifier([input_img])
                                    emotion_output_layer = next(iter(emotion_classifier.outputs))
                                    emotion_predictions = results[emotion_output_layer][0]

                                    # Get the predicted emotion label
                                    emotion_labels = ['neutral', 'happy', 'sad', 'surprise', 'anger']
                                    predicted_emotion_index = np.argmax(emotion_predictions)
                                    emotion = emotion_labels[predicted_emotion_index]
                                    
                                    # Display the emotion on the frame
                                    cv2.putText(cropped_frame, f"Emotion: {emotion}", (10, frame_height - 20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                last_cropped_frame = cropped_frame
                                output_frame = cropped_frame
                            elif last_cropped_frame is not None:
                                output_frame = last_cropped_frame
                            else:
                                output_frame = frame
                        
                    else:
                        output_frame = cv2.resize(frame, (frame_width, frame_height))
                elif isinstance(detector, PersonDetector):
                    # Only detect persons every detection_interval frames
                    if frame_count % detection_interval == 0:
                        persons = detector.detect_persons(frame)
                        last_persons = persons
                    else:
                        persons = last_persons
                    if persons:
                        # Draw green rectangles around detected persons
                        for xmin, ymin, xmax, ymax in persons:
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        if len(persons) != len(last_persons):
                            cv2.putText(frame, f"Detected {len(persons)} persons", (10, frame_height - 20),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        output_frame = frame
                    else:
                        output_frame = cv2.resize(frame, (frame_width, frame_height))
                else:
                    detection_type = "unknown"

                # Calculate actual FPS
                current_time = time.time()
                frame_times.append(current_time)
                # Keep only the last second of frame times
                frame_times = [t for t in frame_times if current_time - t <= 1.0]
                actual_fps = len(frame_times)
                
                # Update FPS display every fps_update_interval frames
                #if frame_count % fps_update_interval == 0:
                cv2.putText(output_frame, f"FPS: {actual_fps}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # For video files, display frame number/total frames
                if isinstance(video_source, str):
                    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    cv2.putText(output_frame, f"Frame: {current_frame}/{total_frames}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                  # Send frame to virtual camera
                if mirror_mode:
                    output_frame = cv2.flip(output_frame, 1)
                if output_frame.shape[1] != frame_width or output_frame.shape[0] != frame_height:
                    output_frame = cv2.resize(output_frame, (frame_width, frame_height))
                
                # Convert to RGB format for virtual camera (pyvirtualcam expects RGB)
                output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                cam.send(output_frame)
                last_frame_time = frame_start_time
                frame_count += 1

    except KeyboardInterrupt:
        print("\nStopping virtual camera...")
    finally:
        cap.release()

def main():
    parser = argparse.ArgumentParser(description="Virtual Camera with Face Detection and Emotion Recognition")
    parser.add_argument("--mirror", type=bool, default=False, help="Enable mirror mode")
    parser.add_argument("--video", type=str, help="Path to video file for input (default: use webcam)", default=None)
    parser.add_argument("--count_mode", type=bool, default=False, help="Count detected faces instead of cropping them")
    parser.add_argument("--target_fps", type=int, default=30, help="Target FPS for the virtual camera")
    parser.add_argument("--detect_object", type=str, default="face", choices=["face", "person", "none"], help="Specify 'face' for face detection or 'person' for person detection")
    parser.add_argument("--detect_emo", type=bool, default=False, help="Enable emotion detection")
    args = parser.parse_args()

    emotion_classifier=None
    if args.detect_emo:
        # Initialize emotion classifier
        emotion_model_xml = ".\\models\\cv\\emotions-recognition-retail-0003\\FP16\\emotions-recognition-retail-0003.xml"
        core = Core()
        emotion_classifier = core.compile_model(emotion_model_xml, 'NPU')
    # Select the appropriate detector based on the --detection_object argument
    if args.detect_object == "face":
        # Initialize face detector
        face_detector = FaceDetector(
            ".\\models\\cv\\face-detection-retail-0004\\FP16-INT8\\face-detection-retail-0004.xml",
            device="NPU",
            confidence_threshold=0.6
        )
        detector = face_detector
        print("Using face detection.")
    elif args.detect_object == "person":
        # Initialize person detector
        person_detector = PersonDetector(
            ".\\models\\cv\\intel\\person-detection-0106\\FP32\\person-detection-0106.xml",
            device="GPU",
            confidence_threshold=0.6
        )
        detector = person_detector
        print("Using person detection.")
    elif args.detect_object == "none":
        detector = None
        print("No detection.")
    else:
        print("Invalid detection type. Please specify 'face' or 'person'.")
        sys.exit(1)

    # Use video file if provided, otherwise use webcam
    video_source = args.video if args.video else 0
    create_virtual_camera(
        detector,
        emotion_classifier,
        video_source=video_source,
        mirror_mode=args.mirror,
        count_mode=args.count_mode,
        target_fps=args.target_fps
    )

if __name__ == "__main__":
    main()