import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
from collections import deque

model = tf.keras.models.load_model("sign_language_model.h5")

dataset_path = "dataset"
class_names = sorted(os.listdir(dataset_path))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

pred_buffer = deque(maxlen=10)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            h, w, _ = frame.shape
            x_list, y_list = [], []

            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)

            padding = 20
            xmin = max(0, xmin - padding)
            ymin = max(0, ymin - padding)
            xmax = min(w, xmax + padding)
            ymax = min(h, ymax + padding)

            hand_img = frame[ymin:ymax, xmin:xmax]

            if hand_img.size != 0:
                img = cv2.resize(hand_img, (224,224))
                img = img / 255.0
                img = np.reshape(img, (1,224,224,3))

                prediction = model.predict(img)
                predicted_class = np.argmax(prediction)

                pred_buffer.append(predicted_class)

                final_class = max(set(pred_buffer), key=pred_buffer.count)

                label = class_names[final_class].replace("_", " ")

                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
                cv2.putText(frame, label, (xmin, ymin-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Sign Language Interpreter", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()