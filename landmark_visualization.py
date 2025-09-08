import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose

# Define body parts and their corresponding colors (in BGR format)
body_parts = {
    'torso': {'connections': [(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
                              (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
                              (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
                              (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP)],
              'color': (255, 255, 0)},
    'left_arm': {'connections': [(mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
                                 (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST)],
                 'color': (0, 255, 0)},
    'right_arm': {'connections': [(mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
                                  (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)],
                  'color': (0, 0, 255)},
    'left_leg': {'connections': [(mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
                                 (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE)],
                 'color': (0, 255, 255)},
    'right_leg': {'connections': [(mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
                                  (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)],
                  'color': (0, 165, 255)}
}
# Define extra connections for feet and head
extra_connections = {
    'head': {'connections': [(mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.RIGHT_EAR),
                             (mp_pose.PoseLandmark.LEFT_EAR, mp_pose.PoseLandmark.NOSE),
                             (mp_pose.PoseLandmark.RIGHT_EAR, mp_pose.PoseLandmark.NOSE)],
             'color': (255, 255, 255)}, # White for the head outline
    'left_foot': {'connections': [(mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_FOOT_INDEX)],
                  'color': (255, 0, 255)}, # Magenta
    'right_foot': {'connections': [(mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)],
                   'color': (255, 0, 255)} # Magenta
}

def draw_perspective_grid(canvas, width, height, color=(40, 40, 40)):
    """Draws a perspective grid with a vanishing point."""
    vanishing_point_x = width // 2
    horizon_y = int(height * 0.45)
    num_lines = 10
    line_thickness = 1
    
    for i in range(num_lines + 1):
        x = int(width * (i / num_lines))
        cv2.line(canvas, (x, height), (vanishing_point_x, horizon_y), color, line_thickness)

    for i in range(1, 10):
        y = horizon_y + int((height - horizon_y) * (i / 10))
        cv2.line(canvas, (0, y), (width, y), color, line_thickness)

def calibrate_z_scale_avg(landmarks_history, known_height_m=1.75):
    """
    Calibrates Z-axis scale by averaging over a history of landmarks.
    """
    heights = []
    for lms in landmarks_history:
        if lms:
            head_y = lms[mp_pose.PoseLandmark.NOSE].y
            foot_y = lms[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y
            relative_height = abs(foot_y - head_y)
            if relative_height > 0:
                heights.append(known_height_m / relative_height)
    if heights:
        return np.mean(heights)
    return 1.0

# Exponential Moving Average filter for smoothing landmark data
class LandmarkEMA:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.smoothed_data = None

    def smooth(self, landmarks):
        if not landmarks:
            return {}

        current_data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        if self.smoothed_data is None:
            self.smoothed_data = current_data
        else:
            self.smoothed_data = self.alpha * current_data + (1 - self.alpha) * self.smoothed_data
        
        smoothed_landmarks = {}
        for idx, (x, y, z) in enumerate(self.smoothed_data):
            smoothed_landmarks[idx] = {'x': x, 'y': y, 'z': z}
        return smoothed_landmarks

# --- Main Program ---

# Replace with the path to your input video file
input_video_path = 'Test 01.mp4'

# --- CALIBRATION SETTINGS ---
person_height_m = 1.75  # <-- ENTER KNOWN HEIGHT IN METERS HERE
calibration_frames = 30  # Number of frames to average height for calibration

# Output video file paths
output_coronal = 'coronal_view_final.mp4'
output_sagittal = 'sagittal_view_final.mp4'
output_transverse = 'transverse_view_final.mp4'

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out_coronal = cv2.VideoWriter(output_coronal, fourcc, fps, (frame_width, frame_height))
out_sagittal = cv2.VideoWriter(output_sagittal, fourcc, fps, (frame_width, frame_height))
out_transverse = cv2.VideoWriter(output_transverse, fourcc, fps, (frame_width, frame_height))

smoother = LandmarkEMA(alpha=0.3)
z_scale = 1.0
landmarks_history = []
frame_count = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        
        canvas_coronal = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        canvas_sagittal = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        canvas_transverse = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        draw_perspective_grid(canvas_coronal, frame_width, frame_height)
        draw_perspective_grid(canvas_sagittal, frame_width, frame_height)
        draw_perspective_grid(canvas_transverse, frame_width, frame_height)

        if results.pose_landmarks:
            landmarks_list = list(results.pose_landmarks.landmark)
            
            # --- Calibration step ---
            if frame_count < calibration_frames:
                landmarks_history.append(landmarks_list)
            elif frame_count == calibration_frames:
                z_scale = calibrate_z_scale_avg(landmarks_history, person_height_m)
                print(f"Calibration successful! Z-scale factor: {z_scale:.2f}")

            smoothed_lms = smoother.smooth(landmarks_list)
            
            # Combine body parts and extra connections for drawing
            all_parts = {**body_parts, **extra_connections}

            # Draw lines first (with glow effect)
            for part_name, part_info in all_parts.items():
                color = part_info['color']
                for connection in part_info['connections']:
                    start_node = connection[0]
                    end_node = connection[1]
                    
                    start_lm = smoothed_lms[start_node]
                    end_lm = smoothed_lms[end_node]
                    
                    avg_z = (start_lm['z'] + end_lm['z']) / 2.0
                    scale_factor = 1.0 / (1.0 - avg_z)
                    
                    # Thicker, darker line for glow effect
                    glow_thickness = int(max(1, 4 * scale_factor))
                    main_thickness = int(max(1, 2 * scale_factor))
                    
                    # 1. Coronal View
                    p1_cor = (int(start_lm['x'] * frame_width), int(start_lm['y'] * frame_height))
                    p2_cor = (int(end_lm['x'] * frame_width), int(end_lm['y'] * frame_height))
                    cv2.line(canvas_coronal, p1_cor, p2_cor, (30, 30, 30), glow_thickness)
                    cv2.line(canvas_coronal, p1_cor, p2_cor, color, main_thickness)

                    # 2. Sagittal View
                    p1_sag = (int(start_lm['z'] * z_scale * 100) + frame_width // 2, int(start_lm['y'] * frame_height))
                    p2_sag = (int(end_lm['z'] * z_scale * 100) + frame_width // 2, int(end_lm['y'] * frame_height))
                    cv2.line(canvas_sagittal, p1_sag, p2_sag, (30, 30, 30), glow_thickness)
                    cv2.line(canvas_sagittal, p1_sag, p2_sag, color, main_thickness)

                    # 3. Transverse View
                    p1_trans = (int(start_lm['x'] * frame_width), int(start_lm['z'] * z_scale * 100) + frame_height // 2)
                    p2_trans = (int(end_lm['x'] * frame_width), int(end_lm['z'] * z_scale * 100) + frame_height // 2)
                    cv2.line(canvas_transverse, p1_trans, p2_trans, (30, 30, 30), glow_thickness)
                    cv2.line(canvas_transverse, p1_trans, p2_trans, color, main_thickness)

            # Draw joints with special styling for head and feet
            for idx, lm in smoothed_lms.items():
                is_head = idx == mp_pose.PoseLandmark.NOSE
                is_foot = idx in [mp_pose.PoseLandmark.LEFT_FOOT_INDEX, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

                joint_size = int(max(2, 6 * (1 / (1 - lm['z']))))

                if is_head or is_foot:
                    joint_color = (0, 255, 255) if is_head else (255, 0, 255)
                    cv2.circle(canvas_coronal, (int(lm['x'] * frame_width), int(lm['y'] * frame_height)), joint_size + 2, (30, 30, 30), -1)
                    cv2.circle(canvas_coronal, (int(lm['x'] * frame_width), int(lm['y'] * frame_height)), joint_size, joint_color, -1)
                else:
                    cv2.circle(canvas_coronal, (int(lm['x'] * frame_width), int(lm['y'] * frame_height)), joint_size, (255, 255, 255), -1)
                
                # Draw joints for other planes (without extra glow)
                cv2.circle(canvas_sagittal, (int(lm['z'] * z_scale * 100) + frame_width // 2, int(lm['y'] * frame_height)), joint_size, (255, 255, 255), -1)
                cv2.circle(canvas_transverse, (int(lm['x'] * frame_width), int(lm['z'] * z_scale * 100) + frame_height // 2), joint_size, (255, 255, 255), -1)
        
        # Write frames to video files
        out_coronal.write(canvas_coronal)
        out_sagittal.write(canvas_sagittal)
        out_transverse.write(canvas_transverse)
        
        cv2.imshow('Coronal View', canvas_coronal)
        cv2.imshow('Sagittal View', canvas_sagittal)
        cv2.imshow('Transverse View', canvas_transverse)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

cap.release()
out_coronal.release()
out_sagittal.release()
out_transverse.release()
cv2.destroyAllWindows()
print("Three enhanced videos saved with seamless skeleton connections and improved quality.")
