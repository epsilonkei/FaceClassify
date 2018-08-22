import os
import cv2
import dlib
import numpy as np
import argparse

# dlib_predictor_path = os.path.expanduser('~')+"/dlib/shape_predictor_68_face_landmarks.dat"
dlib_predictor_path = "./dlib/shape_predictor_68_face_landmarks.dat"


def getFaces(img_path):
    img = cv2.imread(img_path)
    results = []
    height = np.size(img, 0)
    width = np.size(img, 1)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_predictor_path)
    dets = detector(img)
    scale = 0.25
    if len(dets) < 1:
        print ("{0}: Found no face!".format(img_path))
        return results
    for face in dets:
        face = predictor(img, face)
        face = np.array([[face.part(i).x, face.part(i).y] for i in range(68)])
        r_eye_center = np.average(face[36:42], axis=0)
        l_eye_center = np.average(face[42:48], axis=0)
        center = np.array((r_eye_center + l_eye_center) / 2, dtype=np.int32)
        length = np.linalg.norm(r_eye_center - l_eye_center)
        top = center[1] - 10 * int(length*scale)
        top1 = 0 if top < 0 else top
        #
        bottom = center[1] + 10 * int(length*scale)
        bottom1 = height if bottom > height else bottom
        #
        left = center[0] - 10 * int(length*scale)
        left1 = 0 if left < 0 else left
        #
        right = center[0] + 10 * int(length*scale)
        right1 = width if right > width else right
        #
        img_f = np.copy(img[top1: bottom1, left1: right1])
        if top < 0:
            img_f = np.vstack((np.array([img_f[0], ] * (-top)), img_f))
        if bottom > height:
            img_f = np.vstack((img_f, np.array([img_f[-1], ] * (bottom - height))))
        if left < 0:
            img_f = np.hstack((np.transpose(np.array([img_f[:, 0], ] * (-left)), (1, 0, 2)), img_f))
        if right > width:
            img_f = np.hstack((img_f, np.transpose(np.array([img_f[:, -1], ] * (right - width)), (1, 0, 2))))
        results.append(img_f)
    return results


def getFacesWithBorder(img):
    results = []
    bboxes = []
    height = np.size(img, 0)
    width = np.size(img, 1)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_predictor_path)
    dets = detector(img)
    scale = 0.25
    if len(dets) < 1:
        print ("Found no face!")
        return results
    for face in dets:
        face = predictor(img, face)
        face = np.array([[face.part(i).x, face.part(i).y] for i in range(68)])
        r_eye_center = np.average(face[36:42], axis=0)
        l_eye_center = np.average(face[42:48], axis=0)
        center = np.array((r_eye_center + l_eye_center) / 2, dtype=np.int32)
        length = np.linalg.norm(r_eye_center - l_eye_center)
        top = center[1] - 10 * int(length*scale)
        top1 = 0 if top < 0 else top
        #
        bottom = center[1] + 10 * int(length*scale)
        bottom1 = height if bottom > height else bottom
        #
        left = center[0] - 10 * int(length*scale)
        left1 = 0 if left < 0 else left
        #
        right = center[0] + 10 * int(length*scale)
        right1 = width if right > width else right
        #
        img_f = np.copy(img[top1: bottom1, left1: right1])
        if top < 0:
            img_f = np.vstack((np.array([img_f[0], ] * (-top)), img_f))
        if bottom > height:
            img_f = np.vstack((img_f, np.array([img_f[-1], ] * (bottom - height))))
        if left < 0:
            img_f = np.hstack((np.transpose(np.array([img_f[:, 0], ] * (-left)), (1, 0, 2)), img_f))
        if right > width:
            img_f = np.hstack((img_f, np.transpose(np.array([img_f[:, -1], ] * (right - width)), (1, 0, 2))))
        results.append(img_f)
        top2 = center[1] - 6 * int(length*scale)
        top3 = 0 if top2 < 0 else top2
        #
        bottom2 = center[1] + 6 * int(length*scale)
        bottom3 = height if bottom2 > height else bottom2
        #
        left2 = center[0] - 4 * int(length*scale)
        left3 = 0 if left2 < 0 else left2
        #
        right2 = center[0] + 4 * int(length*scale)
        right3 = width if right2 > width else right2
        #
        bboxes.append([top3, bottom3, left3, right3])
    return results, bboxes


def parser_args():
    parser = argparse.ArgumentParser(description='Detect face from image')
    parser.add_argument('--image', '-i', default='images/capture.png', help='Image path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_args()
    # imgs = getFaces(args.image)
    image = cv2.imread(args.image)
    imgs, bboxes = getFacesWithBorder(image)
    for i, box in enumerate(bboxes):
        cv2.rectangle(image, (box[2], box[0]), (box[3], box[1]),
                      (0, 300, 300), thickness=2)
        cv2.imwrite('images/image_result.jpg', image)
