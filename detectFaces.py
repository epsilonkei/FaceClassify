import cv2
import numpy as np
import argparse
from detectAndDraw import load_cascade, load_dlib_predictor


def getFaces(detector, predictor, img_path):
    img = cv2.imread(img_path)
    results = []
    height = np.size(img, 0)
    width = np.size(img, 1)
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


def getFacesWithBorderUsingDlib(detector, predictor, img):
    results = []
    bboxes = []
    height = np.size(img, 0)
    width = np.size(img, 1)
    dets = detector(img)
    scale = 0.25
    if len(dets) < 1:
        # print ("Found no face!")
        return results, bboxes
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


def getFacesWithBorderUsingHaar(cascade, img):
    results = []
    bboxes = []
    height = np.size(img, 0)
    width = np.size(img, 1)
    height, width = img.shape[:2]
    facerect = cascade.detectMultiScale(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.11,
        minNeighbors=3,
        minSize=(100, 100)
    )
    if len(facerect) < 1:
        # print ("Found no face!")
        return results, bboxes
    for rect in facerect:
        org_length = rect[3]/2.
        face_center = np.array((rect[0:2]+rect[2:4]/2.)-np.array([0, org_length*0.25])).astype(np.int32)
        scale = 1.7
        af_length = int(min([min(face_center[0]+org_length*scale, img.shape[1])-face_center[0], face_center[0]-max(face_center[0]-org_length*scale, 0),
                             min(face_center[1]+org_length*scale, img.shape[0])-face_center[1], face_center[1]-max(face_center[1]-org_length*scale, 0)]))
        img_f = np.copy(img[face_center[1]-af_length: face_center[1]+af_length,
                            face_center[0]-af_length: face_center[0]+af_length])
        results.append(img_f)
        top = face_center[1] - int(0.8*af_length)
        bot = face_center[1] + int(0.8*af_length)
        lef = face_center[0] - int(0.6*af_length)
        rig = face_center[0] + int(0.6*af_length)
        bboxes.append([top, bot, lef, rig])
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
    # face_cascade = load_cascade()
    # imgs, bboxes = getFacesWithBorderUsingHaar(face_cascade, image)
    detector, predictor = load_dlib_predictor()
    imgs, bboxes = getFacesWithBorderUsingDlib(detector, predictor, image)
    for i, box in enumerate(bboxes):
        cv2.rectangle(image, (box[2], box[0]), (box[3], box[1]),
                      (0, 300, 300), thickness=2)
        cv2.imwrite('images/image_result.jpg', image)
