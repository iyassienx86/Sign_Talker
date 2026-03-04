import cv2
import time

def run_unit1():
    # Initialize Camera
    cap = cv2.VideoCapture(0)
    
    # Set Resolution (Low = Fast for Pavilion G6)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    p_time = 0 # Previous time for FPS calculation

    print("Unit 1: Testing Camera Eye. Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip for selfie mode
        frame = cv2.flip(frame, 1)

        # Calculate FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        # Display FPS on screen
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), 
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Unit 1 - Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_unit1()