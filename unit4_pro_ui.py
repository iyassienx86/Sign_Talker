import cv2
import mediapipe as mp
import math

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

    word_map = {1: "Please", 2: "I", 3: "Need", 4: "Medical", 5: "Help"}
    full_sentence = "PLEASE I NEED MEDICAL HELP"
    
    is_locked = False
    reached_five = False
    display_text = "Waiting..."

    def get_dist(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # UI Setup - Colors
        color_active = (0, 255, 0)   # Green
        color_locked = (0, 0, 255)   # Red
        current_color = color_locked if is_locked else color_active
        
        if not is_locked:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    lms = hand_lms.landmark
                    fingers = []

                    # AI Logic (Distance Thumb)
                    thumb_dist = get_dist(lms[4], lms[17])
                    fingers.append(1 if thumb_dist > 0.18 else 0)

                    # 4 Fingers Logic
                    for t, m in zip([8, 12, 16, 20], [6, 10, 14, 18]):
                        fingers.append(1 if lms[t].y < lms[m].y else 0)

                    count = fingers.count(1)

                    if count > 0:
                        display_text = word_map.get(count, "...")
                        if count == 5: reached_five = True 
                    elif count == 0 and reached_five:
                        display_text = full_sentence
                        is_locked = True 
                    elif count == 0:
                        display_text = "Ready..."

                    # Draw hand connections with thinner lines to keep view clear
                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                                         mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1),
                                         mp_draw.DrawingSpec(color=current_color, thickness=2, circle_radius=1))
                    
                    # TOP STATUS BAR (Improved visibility)
                    cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
                    cv2.putText(frame, f"{hand_label.upper()} HAND | FINGERS: {count}", (20, 35), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # --- 4. NEW TRANSPARENT UI OVERLAY ---
        # Draw a semi-transparent overlay at the bottom
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-80), (w, h), (0, 0, 0), -1)
        alpha = 0.4  # Transparency factor
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Main Word Display (Bottom Center)
        text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
        text_x = (w - text_size[0]) // 2
        
        # Add a subtle glow/shadow to text for readability
        cv2.putText(frame, display_text, (text_x + 2, h - 33), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(frame, display_text, (text_x, h - 35), cv2.FONT_HERSHEY_DUPLEX, 1.2, current_color, 2)

        # Red border if Locked
        if is_locked:
            cv2.rectangle(frame, (0, 0), (w, h), color_locked, 10)

        cv2.imshow("Sign Talker - UI Refresh", frame)
        
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