import cv2
import numpy as np
import sys
import time 
from time import gmtime, strftime

CAPSIZEX = 1280
CAPSIZEY = 720

DSPSIZEX = 480
DSPSIZEY = 320

blocksize = 11

def gstreamer_pipeline(
    capture_width = CAPSIZEX,
    capture_height = CAPSIZEY,
    display_width = DSPSIZEX,
    display_height = DSPSIZEY,
    framerate = 30,
    flip_method = 0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )



def show_camera():


    # Флаг для отображения оригинального или преобразованного изображения
    flag = 1

    # Подсчет FPS
    prev_frame_time = 0
    new_frame_time = 0

    fps_timer = 0
    frame_counter = 0

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=4))

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=4), cv2.CAP_GSTREAMER)

    # Window
    while (cap.isOpened()):

        ret_val, frame = cap.read()

        prev_frame_time = new_frame_time
        new_frame_time = time.time()

        current_frame_time = new_frame_time - prev_frame_time
        frame_counter = frame_counter + 1
        fps_timer = fps_timer + current_frame_time

        if frame_counter == 30:
            fps = int(1 / (fps_timer / 30))
            frame_counter = 0
            fps_timer = 0

        # Применение бинаризации
        if flag < 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, 2)

        # Write fps
        cv2.putText(frame, fps, (7, 35), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 9, cv2.LINE_AA)
        cv2.putText(frame, fps, (7, 35), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 3, cv2.LINE_AA)

        # Show video
        cv2.imshow('Cam', frame)

        # This also acts as
        keyCode = cv2.waitKey(1) & 0xFF

        # Скриншот изображения
        if keyCode == 201:

            if flag < 0:
                screenshot_name = "Binarized_"
            else:
                screenshot_name = "Original_"

            screenshot_name = screenshot_name + str(blocksize) + "BS_" + strftime("%Y-%m-%d_%H:%M:%S", gmtime()) + '.png'

            cv2.imwrite(screenshot_name, frame)
            print("Image saved as: " + screenshot_name)

        # Изменение выводимого изображения
        if keyCode == 32:
            flag = -flag

        # Stop the program on the ESC key
        if keyCode == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if sys.argv[1]:
        blocksize = int(sys.argv[1])
    show_camera()