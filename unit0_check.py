import cv2
import mediapipe as mp
import sys

def zero_point_check():
    print("--- UNIT 0: INITIALIZATION START ---")
    
    # Check 1: MediaPipe Module
    try:
        test_hands = mp.solutions.hands.Hands()
        print("[OK] MediaPipe Light: Loaded")
    except Exception as e:
        print(f"[FAIL] MediaPipe Error: {e}")
        return False

    # Check 2: Camera Hardware
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("[OK] Camera Hardware: Active & Capturing")
        else:
            print("[FAIL] Camera: Access denied or in use")
        cap.release()
    else:
        print("[FAIL] Camera: Not detected")
        return False

    print("--- STATUS: BASELINE STABLE. READY FOR REPO PUSH ---")
    return True

if __name__ == "__main__":
    if not zero_point_check():
        sys.exit(1)