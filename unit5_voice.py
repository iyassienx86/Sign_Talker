import cv2
import mediapipe as mp
import math
import os
import time
import threading # NEW: Prevents the video from freezing
from gtts import gTTS 

def run_unit5_final():
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
    has_spoken = False 
    display_text = "Waiting..."

    def get_dist(p1, p2):
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    # Helper function for background audio
    def play_audio_async(file_path):
        os.system(f"mpg123 -q {file_path}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        color_active = (0, 255, 0)
        color_locked = (0, 0, 255)
        current_color = color_locked if is_locked else color_active
        
        if not is_locked:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    lms = hand_lms.landmark
                    fingers = []

                    # THE PERFECT AI LOGIC
                    thumb_dist = get_dist(lms[4], lms[17])
                    fingers.append(1 if thumb_dist > 0.18 else 0)

                    for t, m in zip([8, 12, 16, 20], [6, 10, 14, 18]):
                        fingers.append(1 if lms[t].y < lms[m].y else 0)

                    count = fingers.count(1)

                    if count > 0:
                        display_text = word_map.get(count, "...")
                        if count == 5: reached_five = True 
                    elif count == 0 and reached_five:
                        display_text = full_sentence
                        is_locked = True 

                    mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                    
                    # Top Info Bar
                    cv2.rectangle(frame, (0, 0), (w, 50), (40, 40, 40), -1)
                    cv2.putText(frame, f"{hand_label.upper()} | FINGERS: {count}", (20, 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --- 2. THE HUMAN VOICE TRIGGER (THREADED - NO FREEZE) ---
        if is_locked and not has_spoken:
            print("Generating Human Voice...")
            try:
                # 1. Generate audio file
                tts = gTTS(text=full_sentence, lang='en', slow=False)
                tts.save("help.mp3")
                
                # 2. Use Threading to play sound in the background
                # This prevents the camera 'while' loop from stopping
                audio_thread = threading.Thread(target=play_audio_async, args=("help.mp3",))
                audio_thread.start()
                
                has_spoken = True 
                print("Voice thread started successfully!")
            except Exception as e:
                print(f"Audio Error: {e}")
                has_spoken = True 

        # --- 3. UI RENDER ---
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-80), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        t_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        cv2.putText(frame, display_text, ((w - t_size[0]) // 2, h - 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, current_color, 3)

        if is_locked:
            cv2.rectangle(frame, (0, 0), (w, h), color_locked, 15)

        cv2.imshow("Sign Talker Final", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('r'):
            is_locked = False
            reached_five = False
            has_spoken = False
            display_text = "Waiting..."
            # Note: We don't delete the file immediately because the thread might still be playing it

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_unit5_final()