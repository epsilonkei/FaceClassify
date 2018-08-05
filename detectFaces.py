import os
import cv2
import dlib
import numpy as np

# dlib_predictor_path = os.path.expanduser('~')+"/dlib/shape_predictor_68_face_landmarks.dat"
dlib_predictor_path = "./dlib/shape_predictor_68_face_landmarks.dat"


def getFaces(img):
    results = []
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
        top = center[1] - 11 * int(length*scale)
        top1 = 0 if top < 0 else top

        bottom = center[1] + 11 * int(length*scale)
        bottom1 = height if bottom > height else bottom

        left = center[0] - 9 * int(length*scale)
        left1 = 0 if left < 0 else left

        right = center[0] + 9 * int(length*scale)
        right1 = width if right > width else right

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
        #results.append(img[top : bottom, left : right])

    return results


if __name__ == '__main__':
    lena = getFaces(cv2.imread('images/capture.png'))
    for i, img in enumerate(lena):
        #img = cv2.resize(img, (72, 88))
        cv2.imwrite('images/image{0}.jpg'.format(i), img)
