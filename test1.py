import cv2
import numpy as np

# from PIL import Image
# from PIL import ImageEnhance


# def threshold(filename, step=20):
#     img    = Image.open(filename).convert("L")
#     img = ImageEnhance.Contrast(img).enhance(1.2)
#     pixels = list(img.getdata())
#     arr    = np.array(pixels)
#     arr2d  = arr.reshape(img.size)

#     blocks = np.reshape(arr2d, (-1, step, step))
#     for block in blocks:
#         factor = 1.2 # CHANGE DEPENDING ON RESULT -> Need algorithm
#         mean   = np.mean(block)
#         thresh = mean/factor

#         block[block <= thresh] = 0
#         block[block > thresh] = 1

#     arr2d = np.reshape(blocks, (1, -1))

#     img2  = Image.new("1", img.size)
#     img2.putdata(arr2d[0].tolist())
#     img2.save("test.jpg")
#     img2.show()



#     return img2



def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=480,
    display_height=320,
    framerate=30,
    flip_method=0,
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
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=4))

    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=4), cv2.CAP_GSTREAMER)

    # Window
    while (cap.isOpened()):

        ret_val, frame = cap.read()

        invert = ~frame

        print(frame)

        # Show video
        cv2.imshow('Original', frame)
        cv2.imshow('Inverted', invert)

        # This also acts as
        keyCode = cv2.waitKey(1) & 0xFF

        # Stop the program on the ESC key
        if keyCode == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_camera()