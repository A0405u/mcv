import cv2
import numpy as np
import sys

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

    flag = 1
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=4))

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=4), cv2.CAP_GSTREAMER)

    # Window
    while (cap.isOpened()):

        ret_val, frame = cap.read()

        #binary = threshold(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, 2)

        # Show video
        if flag > 0:
            cv2.imshow('Original', frame)
        else:
            cv2.imshow('Binary', binary)

        # This also acts as
        keyCode = cv2.waitKey(1) & 0xFF

        if keyCode == 32:
            flag = -flag
            print(flag)

        # Stop the program on the ESC key
        if keyCode == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if sys.argv[1]:
        blocksize = int(sys.argv[1])
    show_camera()