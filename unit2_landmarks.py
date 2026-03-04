import cv2
import mediapipe as mp
import time

def run_unit2():
    cap = cv2.VideoCapture(0)
    
    # 1. Initialize MediaPipe "Brain"
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    
    # Configuration for CPU Efficiency on HP Pavilion g6
    hands = mp_hands.Hands(
        static_image_mode=False,    # Treat as video, not separate images
        max_num_hands=1,            # Only track one hand to save CPU
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.5
    )

    p_time = 0

    print("Unit 2: Landmark Extraction. Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # Mirror the image
        frame = cv2.flip(frame, 1)
        
        # 2. AI Processing (MediaPipe requires RGB, OpenCV uses BGR)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # 3. If a hand is detected, draw the 21 dots
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                )

        # FPS Calculation to monitor CPU load
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), 
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow("Unit 2 - Landmark Brain", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_unit2()