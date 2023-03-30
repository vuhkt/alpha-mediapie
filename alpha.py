import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
kb = Controller()
def t_rex():

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
          for hand_landmarks in results.multi_hand_landmarks:
            if hand_landmarks.landmark[12].y < hand_landmarks.landmark[11].y and hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y and hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y:
                kb.press(Key.space)
            else:
                kb.release((Key.space))
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()
def dua_xe():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            max_num_hands=2,  # số lượng bàn tay tối đa
            model_complexity=0,  # độ phức tạp của mốc bàn tay giá trị từ 0 đênns 1
            min_detection_confidence=0.5, # độ tin cậy tối thiêu để phát hiện ra là bàn tay giá trị từ 0 đến 1
            min_tracking_confidence=0.5) as hands:  # độ tin cậy tối thiểu để theo dõi các mốc bàn tay từ 0 đên 1 giá trị càng cao tính mạnh mẽ của giả pháp càng mạnh và đỗ trể càng lón
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    print(hand_landmarks.landmark[12].x, '     ', hand_landmarks.landmark[12].y)
                    if hand_landmarks.landmark[0].x >= 0.5 and (
                            hand_landmarks.landmark[12].y > hand_landmarks.landmark[11].y and \
                            hand_landmarks.landmark[16].y > hand_landmarks.landmark[15].y and hand_landmarks.landmark[
                                8].y < hand_landmarks.landmark[7].y):
                        kb.release(Key.up)
                        kb.release(Key.down)
                        kb.press(Key.space)

                    elif hand_landmarks.landmark[0].x >= 0.5 and (
                            hand_landmarks.landmark[12].y < hand_landmarks.landmark[11].y and \
                            hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y and hand_landmarks.landmark[
                                8].y < hand_landmarks.landmark[7].y):
                        kb.release(Key.down)
                        kb.release(Key.space)
                        kb.press(Key.up)
                    elif hand_landmarks.landmark[0].x >= 0.5 and (
                            hand_landmarks.landmark[12].y > hand_landmarks.landmark[11].y and \
                            hand_landmarks.landmark[16].y > hand_landmarks.landmark[15].y and hand_landmarks.landmark[
                                8].y > hand_landmarks.landmark[7].y):
                        kb.release(Key.up)
                        kb.release(Key.space)
                        kb.press(Key.down)
            else:
                kb.release(Key.up)
                kb.release(Key.space)
                kb.release(Key.down)

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

