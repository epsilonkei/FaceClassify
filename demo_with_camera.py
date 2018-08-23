import argparse
import cv2
import detectFaces
from classify import classifyWithImgResult


def parser_args():
    parser = argparse.ArgumentParser(description='Face Classify Demo using camera')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dlib', '-d', default=1, type=int,
                        help='Using dlib or not (positive value for using Dlib, \
                        negative value for using Haar)')
    args = parser.parse_args()
    return args


def demo_with_camera():
    args = parser_args()
    cap = cv2.VideoCapture(0)
    if args.dlib >= 0:
        detector, predictor = detectFaces.load_dlib_predictor()
    else:
        cascade = detectFaces.load_cascade()
    while True:
        ret, frame = cap.read()
        orgHeight, orgWidth = frame.shape[:2]
        frame = frame[:, ::-1]
        while (orgWidth > 1000 or orgHeight > 1000):
            frame = cv2.resize(frame, (int(orgWidth/2), int(orgHeight/2)))
            orgHeight, orgWidth = frame.shape[:2]
        if args.dlib >= 0:
            f_imgs, bboxes = detectFaces.getFacesWithBorderUsingDlib(detector, predictor, frame)
        else:
            f_imgs, bboxes = detectFaces.getFacesWithBorderUsingHaar(cascade, frame)
        img = classifyWithImgResult(args.gpu, frame, f_imgs, bboxes)
        cv2.imshow("Face Classify", img)
        k = cv2.waitKey(1)  # Wait for 1msec
        if k == 27:  # ESC key to quit
            break
    cap.release()
    cv2.destroyWindow("Face Classify")


if __name__ == '__main__':
    demo_with_camera()
