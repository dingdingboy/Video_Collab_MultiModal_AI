import cv2
import numpy as np

def calculate_frame_similarity(frame1, frame2):
    """
    Calculates the similarity between two frames using Mean Squared Error (MSE).
    Lower MSE indicates higher similarity.
    """
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame1 = cv2.resize(frame1, (224, 224))
    frame2 = cv2.resize(frame2, (224, 224))
    mse = np.mean((frame1 - frame2) ** 2)
    return mse

def analyze_video_similarity(video_path1, video_path2):
    """
    Analyzes the content similarity between two video files by comparing all frames.

    Args:
        video_path1 (str): Path to the first video file.
        video_path2 (str): Path to the second video file.

    Returns:
        float: The average Mean Squared Error (MSE) between all corresponding frames of the two videos.
               Lower MSE indicates higher similarity. Returns -1 if there is an error.
    """
    try:
        cap1 = cv2.VideoCapture(video_path1)
        cap2 = cv2.VideoCapture(video_path2)

        if not cap1.isOpened() or not cap2.isOpened():
            print("Error: Could not open one or both video files.")
            return -1

        frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

        min_frame_count = min(frame_count1, frame_count2)
        
        if min_frame_count == 0:
            print("Error: One or both videos have no frames.")
            return -1

        total_similarity = 0.0
        for frame_number in range(min_frame_count):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                print(f"Error: Could not read frame {frame_number} from one or both video files.")
                break  # Exit the loop if a frame cannot be read

            similarity_score = calculate_frame_similarity(frame1, frame2)
            total_similarity += similarity_score

        cap1.release()
        cap2.release()

        average_similarity = total_similarity / min_frame_count if min_frame_count > 0 else -1
        return average_similarity

    except Exception as e:
        print(f"An error occurred: {e}")
        return -1

if __name__ == '__main__':
    # Example usage:
    video1_path = '.\\.cache\\video1.mp4'  # Replace with your video file path
    video2_path = '.\\.cache\\desktop_lasttime.mp4'  # Replace with your video file path

    similarity = analyze_video_similarity(video1_path, video2_path)

    if similarity != -1:
        print(f"Average similarity between the two videos: {similarity}")