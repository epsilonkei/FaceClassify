import cv2
import numpy as np
import fcn


def draw_instance_bboxes(img, bboxes, captions=None,
                         thickness=1, alpha=0.5, draw=None):
    '''
    img: input image,
    bboxes: bounding box list (np.ndarray),
    captions: caption list
    '''
    # validation
    assert isinstance(img, np.ndarray)
    assert img.shape == (img.shape[0], img.shape[1], 3)
    assert img.dtype == np.uint8
    bboxes = np.asarray(bboxes)
    assert isinstance(bboxes, np.ndarray)
    assert bboxes.shape == (bboxes.shape[0], 4)
    if draw is None:
        draw = [True] * bboxes.shape[0]
    else:
        assert len(draw) == bboxes.shape[0]

    if captions is not None:
        captions = np.asarray(captions)
        assert isinstance(captions, np.ndarray)
        assert captions.shape[0] == bboxes.shape[0]

    img_viz = img.copy()
    cmap = fcn.utils.labelcolormap(bboxes.shape[0])

    CV_AA = 16
    for i_box in range(bboxes.shape[0]):
        if not draw[i_box]:
            continue
        box = bboxes[i_box]
        top, bot, lef, rig = box.astype(int).tolist()
        color = cmap[i_box]
        color = (color * 255).tolist()
        cv2.rectangle(img_viz, (lef, top), (rig, bot), color[::-1],
                      thickness=thickness, lineType=CV_AA)

        if captions is not None:
            caption = captions[i_box]
            font_scale = 0.5
            lines = caption.split('\n')
            for i, line in enumerate(lines):
                ret, baseline = cv2.getTextSize(
                    line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
                cv2.putText(img_viz, line, (lef, bot - (3*(len(lines) - i) - 2) * baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                            1, CV_AA)

    return img_viz
