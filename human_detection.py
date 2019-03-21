import cv2
import numpy as np
import mix_Gaussian_module as mGm
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("level_4.mp4")
size = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 3

last_frame = np.zeros(size).astype(np.float)

frame_id = 0
while True:
    frame_id = frame_id + 1         # frame counter

    ret, frame = cap.read()
    if not ret:
        break

    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float)    # convert to gray picture
    frame = frame.astype(np.float)
    cv2.imshow("input", frame.astype(np.uint8))
    cv2.imshow("last_frame", last_frame.astype(np.uint8))
    blur_frame = cv2.blur(frame, (5, 5))

    # motion detection method
    if frame_id == 1:
        last_frame = blur_frame

    dynamic_item = np.abs(blur_frame - last_frame)
    dynamic_item = np.max(dynamic_item, axis=2)
    print(dynamic_item.shape)
    dynamic_item = np.where(dynamic_item > 10, 255, 0)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dynamic_item = cv2.morphologyEx(src=dynamic_item.astype(np.uint8), op=cv2.MORPH_OPEN, kernel=k)
    # dynamic_item = cv2.morphologyEx(src=dynamic_item.astype(np.uint8), op=cv2.MORPH_CLOSE, kernel=k)

    print(dynamic_item.max(), dynamic_item.min(), dynamic_item.shape)

    cv2.imshow("result", dynamic_item.astype(np.uint8))
    alpha = 0.8
    last_frame = blur_frame*alpha + last_frame*(1 - alpha)

    # background  detection method

    cv2.imwrite("a.jpg", dynamic_item)


    if cv2.waitKey(0) == 27:
        break

cap.release()
