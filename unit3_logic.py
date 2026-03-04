import cv2
import mediapipe as mp

def run_unit3():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    
    # Optimized for HP Pavilion g6 performance
    hands = mp_hands.Hands(
        model_complexity=0, 
        max_num_hands=1, 
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.5
    )
    mp_draw = mp.solutions.drawing_utils

    # Aligned Dictionary: 1 finger = Please, 2 = I, etc.
    words = {
        0: "Ready...", 
        1: "Please", 
        2: "I", 
        3: "Need", 
        4: "Medical", 
        5: "Help"
    }

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1) # Mirroring for user comfort
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # We use zip to get both the Landmarks AND the Label (Left/Right)
            for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label # "Left" or "Right"
                landmarks = hand_lms.landmark
                fingers = []

                # --- 1. UNIVERSAL THUMB LOGIC (FIXED) ---
                # On a mirrored Right hand, thumb is OPEN if X is LESS than Index Knuckle
                # On a mirrored Left hand, thumb is OPEN if X is GREATER than Index Knuckle
                if hand_label == "Right":
                    if landmarks[4].x < landmarks[5].x:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else: # Left Hand
                    if landmarks[4].x > landmarks[5].x:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # --- 2. FOUR FINGERS LOGIC (VERTICAL) ---
                tips = [8, 12, 16, 20]
                knuckles = [6, 10, 14, 18]
                
                for i in range(4):
                    if landmarks[tips[i]].y < landmarks[knuckles[i]].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # --- 3. DISPLAY RESULTS ---
                total_fingers = fingers.count(1)
                text = words.get(total_fingers, "...")

                cv2.putText(frame, f"{hand_label}: {text}", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Unit 3 - Universal Hand Logic", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_unit3()