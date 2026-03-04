import cv2
import mediapipe as mp

def run_unit3():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    # Complexity 0 is the "Lite" version for your HP Pavilion
    hands = mp_hands.Hands(model_complexity=0, max_num_hands=1) 
    mp_draw = mp.solutions.drawing_utils

    # Our 5-word Dictionary
    words = {1: "Please", 2: "I", 3: "Need", 4: "Medical", 5: "Help"}

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                landmarks = hand_lms.landmark
                fingers = []

                # 1. Thumb Logic (Special: Horizontal check)
                if landmarks[4].x > landmarks[3].x:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # 2. Four Fingers Logic (Vertical check: Tip vs Knuckle)
                finger_tips = [8, 12, 16, 20]
                finger_knuckles = [6, 10, 14, 18]
                
                for i in range(4):
                    if landmarks[finger_tips[i]].y < landmarks[finger_knuckles[i]].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # 3. Count total 1s and get the word
                total_fingers = fingers.count(1)
                text = words.get(total_fingers, "Waiting...")

                # Display the word
                cv2.putText(frame, text, (200, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)

                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Unit 3 - Finger Logic", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_unit3()