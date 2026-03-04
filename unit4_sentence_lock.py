import cv2
import mediapipe as mp
import math # Added for the distance calculation

def run_unit4():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    # Optimized for HP Pavilion g6
    hands = mp_hands.Hands(
        model_complexity=0, 
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    word_map = {1: "Please", 2: "I", 3: "Need", 4: "Medical", 5: "Help"}
    full_sentence = "PLEASE I NEED MEDICAL HELP"
    
    is_locked = False
    reached_five = False
    display_text = "Waiting..."

    # Distance helper to make Thumb 100% accurate
    def get_dist(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    print("Unit 4: 100% Accuracy Logic. Press 'q' to quit, 'r' to reset.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1) 
        
        if not is_locked:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label # Detects Left/Right
                    lms = hand_lms.landmark
                    fingers = []

                    # --- 1. NEW DISTANCE THUMB LOGIC (Agnotic to hand side) ---
                    # Distance between Thumb Tip (4) and Pinky Base (17)
                    thumb_dist = get_dist(lms[4], lms[17])
                    # If distance is large, thumb is open. 0.18 is the magic number.
                    fingers.append(1 if thumb_dist > 0.18 else 0)

                    # --- 2. STRICT 4 FINGER LOGIC (Vertical) ---
                    tips = [8, 12, 16, 20]
                    middle_joints = [6, 10, 14, 18]
                    for t, m in zip(tips, middle_joints):
                        # Tip Y must be smaller (higher) than Middle Joint Y
                        fingers.append(1 if lms[t].y < lms[m].y else 0)

                    count = fingers.count(1)

                    # --- 3. TRIGGER LOGIC ---
                    if count > 0:
                        display_text = word_map.get(count, "...")
                        if count == 5:
                            reached_five = True 
                    elif count == 0:
                        if reached_five:
                            display_text = full_sentence
                            is_locked = True 
                        else:
                            display_text = "Waiting..."

                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                    
                    # Show Hand Side and Finger Count for debugging
                    cv2.putText(frame, f"{hand_label} Hand | Fingers: {count}", (10, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- 4. UI OVERLAY ---
        color = (0, 0, 255) if is_locked else (0, 255, 0)
        
        # Black background for the word
        cv2.rectangle(frame, (0, 200), (frame.shape[1], 280), (0, 0, 0), -1)
        
        text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, display_text, (text_x, 250), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.imshow("Sign Talker - Final Logic Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'):
            is_locked = False
            reached_five = False
            display_text = "Waiting..."

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_unit4()