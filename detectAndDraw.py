import cv2
import numpy as np
import os
import argparse
import dlib

dlib_predictor_path = "./dlib/shape_predictor_68_face_landmarks.dat"


def detectAndDrawWithHaar(frame, save_path):
    cascade_path = os.path.expanduser('~')+"/.pyenv/versions/anaconda3-4.3.0/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    orgHeight, orgWidth = frame.shape[:2]
    while (orgWidth > 1000 or orgHeight > 1000):
        frame = cv2.resize(frame, (int(orgWidth/2), int(orgHeight/2)))
        orgHeight, orgWidth = frame.shape[:2]
    facerect = cascade.detectMultiScale(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.11,
        minNeighbors=3,
        minSize=(100, 100)
    )
    img = frame.copy()
    if len(facerect) != 0:
        for rect in facerect:
            org_length = rect[3]/2.
            face_center = np.array((rect[0:2]+rect[2:4]/2.)-np.array([0, org_length*0.25])).astype(np.int32)
            scale = 1.7
            af_length = int(min([min(face_center[0]+org_length*scale, img.shape[1])-face_center[0], face_center[0]-max(face_center[0]-org_length*scale, 0),
                                 min(face_center[1]+org_length*scale, img.shape[0])-face_center[1], face_center[1]-max(face_center[1]-org_length*scale, 0)]))
            cv2.rectangle(
                img,
                tuple(face_center-af_length),
                tuple(face_center+af_length),
                (0, 300, 300),
                thickness=2)
    cv2.imwrite(save_path, img)


def detectAndDrawWithDlib(frame, save_path):
    results = []
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_predictor_path)
    orgHeight, orgWidth = frame.shape[:2]
    while (orgWidth > 1000 or orgHeight > 1000):
        frame = cv2.resize(frame, (int(orgWidth/2), int(orgHeight/2)))
        orgHeight, orgWidth = frame.shape[:2]
    dets = detector(frame)
    scale = 0.25
    if len(dets) < 1:
        print ("Found no face!")
        return results
    for face in dets:
        face = predictor(frame, face)
        face = np.array([[face.part(i).x, face.part(i).y] for i in range(68)])
        for i in range(face.shape[0]):
            cv2.circle(frame, (int(face[i][0]), int(face[i][1])), 1, (0, 0, 255), 3)
        r_eye_center = np.average(face[36:42], axis=0)
        l_eye_center = np.average(face[42:48], axis=0)
        center = np.array((r_eye_center + l_eye_center) / 2, dtype=np.int32)
        length = np.linalg.norm(r_eye_center - l_eye_center)
        top = center[1] - 11 * int(length*scale)
        top1 = 0 if top < 0 else top
        #
        bottom = center[1] + 11 * int(length*scale)
        bottom1 = orgHeight if bottom > orgHeight else bottom
        #
        left = center[0] - 9 * int(length*scale)
        left1 = 0 if left < 0 else left
        #
        right = center[0] + 9 * int(length*scale)
        right1 = orgWidth if right > orgWidth else right
        #
        cv2.rectangle(frame, (left1, top1), (right1, bottom1),
                      (0, 300, 300), thickness=2)
        cv2.imwrite(save_path, frame)


if __name__ == '__main__':
    save_path_haar = 'images/Haar.png'
    save_path_dlib = 'images/dlib.png'
    parser = argparse.ArgumentParser(description='Face Classify from image')
    parser.add_argument('--image', '-i', default='images/capture.png', help='Image path')
    args = parser.parse_args()
    detectAndDrawWithHaar(cv2.imread(args.image), save_path_haar)
    detectAndDrawWithDlib(cv2.imread(args.image), save_path_dlib)
