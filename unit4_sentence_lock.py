import cv2
import mediapipe as mp

def run_unit4():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity=0, 
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    # Word Mapping
    word_map = {1: "Please", 2: "I", 3: "Need", 4: "Medical", 5: "Help"}
    full_sentence = "PLEASE I NEED MEDICAL HELP"
    
    is_locked = False
    reached_five = False
    display_text = "Waiting for signal..."

    print("Unit 4: Final Statement. Press 'q' to quit, 'r' to reset.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        
        # 1. PROCESS ONLY IF NOT LOCKED
        if not is_locked:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    lms = hand_lms.landmark
                    fingers = []

                    # Thumb Logic (Universal)
                    if hand_label == "Right":
                        fingers.append(1 if lms[4].x < lms[5].x else 0)
                    else:
                        fingers.append(1 if lms[4].x > lms[5].x else 0)

                    # 4 Fingers Logic
                    tips = [8, 12, 16, 20]
                    knuckles = [6, 10, 14, 18]
                    for t, k in zip(tips, knuckles):
                        fingers.append(1 if lms[t].y < lms[k].y else 0)

                    count = fingers.count(1)

                    # LOGIC: If we see fingers, update current word
                    if count > 0:
                        display_text = word_map.get(count, "...")
                        if count == 5:
                            reached_five = True # Arm the trigger
                    
                    # TRIGGER: If we see a FIST (0) AFTER seeing 5 fingers -> LOCK
                    elif count == 0 and reached_five:
                        display_text = full_sentence
                        is_locked = True 

                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        # --- 2. UI OVERLAY ---
        color = (0, 0, 255) if is_locked else (0, 255, 0)
        status = "LOCKED" if is_locked else "ACTIVE"
        
        # Status Box
        cv2.rectangle(frame, (0, 0), (220, 40), (0, 0, 0), -1)
        cv2.putText(frame, f"STATUS: {status}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Main Text (Center of screen)
        text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, display_text, (text_x, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.imshow("Unit 4 - Final Statement", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        if key == ord('r'): # Press 'r' to unlock and start over
            is_locked = False
            reached_five = False
            display_text = "Waiting..."

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_unit4() # FIXED: Called the correct function name!