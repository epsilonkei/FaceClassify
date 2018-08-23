from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import dlib
import cv2

dlib_predictor_path = "./dlib/shape_predictor_68_face_landmarks.dat"


def parser_args():
    ap = argparse.ArgumentParser(description='Capture face demo using Dlib library')
    ap.add_argument("--shape-predictor", "-p", default=dlib_predictor_path,
                    help="path to facial landmark predictor")
    args = ap.parse_args()
    return args


def capture_face_with_dlib(save_path):
    args = parser_args()
    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)
    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] camera sensor warming up...")
    vs = VideoStream().start()

    while True:
        frame = vs.read()
        orgHeight, orgWidth = frame.shape[:2]
        while (orgWidth > 1000 or orgHeight > 1000):
            frame = imutils.resize(frame, (int(orgWidth/2), int(orgHeight/2)))
            orgHeight, orgWidth = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        img = frame.copy()
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            for (x, y) in shape:
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        cv2.imshow("Frame", img[:, ::-1])
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key to quit
            cv2.imwrite(save_path, frame[:, ::-1])
            break

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    save_path = 'images/capture.png'
    capture_face_with_dlib(save_path)
