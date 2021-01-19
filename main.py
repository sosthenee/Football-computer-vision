import numpy as np
import cv2

import random as rng
import tensorflow as tf
from collections import deque

from load_model_data import load
from object_detection.utils import visualization_utils as viz_utils

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# Note: Model used for SSDLite_Mobilenet_v2


detection_model, categories, category_index = load()

rng.seed(12345)


def thresh_callback(src):
    threshold = 190

    src_temp = src.copy()
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.blur(src_gray, (3, 3))
    ## [Canny]
    # Detect edges using Canny
    canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
    ## [Canny]
    # cv2.imshow('Contours', canny_output)
    # cv2.waitKey(0)

    ## [findContours]
    # Find contours
    (contours, _) = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ## [findContours]

    ## [allthework]
    # Approximate contours to polygons + get bounding rects and circles
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

    ## [allthework]

    percentage = [0] * 200

    ball = 0
    for n in range(len(contours)):

        xmin_ball = int(boundRect[n][0])
        ymin_ball = int(boundRect[n][1])
        w_ball = int(boundRect[n][2])
        h_ball = int(boundRect[n][3])

        if 25 >= w >= 12 and 25 >= h >= 12 and ymin > 200:
            crop_img = src_temp[ymin_ball:ymin_ball + h_ball + 5, xmin_ball:xmin_ball + w_ball + 5]
            # cv2.imshow('Contours', crop_img)
            # cv2.waitKey(0)

            color, ratio = detect_ball(crop_img)
            # print(percentage)

            if color == 'ball':
                percentage[n] = ratio
                # print("color" , color)

        if any(percentage):
            ball = np.argmax(percentage)
            centers[ball] = [round(x) for x in centers[ball]]

            # cv2.rectangle(src, (int(boundRect[ball][0]), int(boundRect[ball][1])), \
            #              (int(boundRect[ball][0] + boundRect[ball][2]), int(boundRect[ball][1] + boundRect[ball][3])),
            #              (0, 0, 255), 1)
            cv2.circle(src, (int(centers[ball][0]), int(centers[ball][1])), int(radius[ball]), (0, 0, 255), 2)

    # print("center", centers[ball])
    if ball != 0:
        pts.appendleft(centers[ball])
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # todo : wip trying to draw ball last positions
        #otherwise, compute the thickness of the line and
        # draw the connecting lines
        #thickness = int(np.sqrt(5 / float(i + 1)) * 2.5)
        # print("thickness",thickness)
        # cv2.line(src, tuple(pts[i - 1]), tuple(pts[i]), (0, 0, 255), thickness)

        # cv2.imshow('Contours', crop_img)
        # cv2.waitKey(0)

        ## [forContour]
        # Draw polygonal contour + bonding rects + circles


        # print("oui")
        # color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))

        # cv.drawContours(src, contours_poly, i, color)

        # imS = cv2.resize(src, (960, 540))
        # cv2.imshow('Contours', imS)
        # print("percentage " ,np.argmax(percentage))
        # cv2.waitKey(0)

        # cv.circle(src, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
        ## [forContour]


def count_nonblack_np(img):
    """Return the number of pixels in img that are not black.
    img must be a Numpy array with colour values along the last axis.

    """
    return img.any(axis=-1).sum()


def detect_ball(image):
    # define the list of boundaries
    boundaries = [
        ([0, 0, 0], [179, 70, 255]),  # white
        ([36, 52, 0], [60, 255, 140])  # green
    ]
    i = 0
    green = 0
    c_white = False
    c_green = False
    for (lower, upper) in boundaries:

        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        # print("lower" , lower)

        green = 0
        # find the colors within the specified boundaries and apply
        # the mask
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)
        tot_pix = count_nonblack_np(image)
        color_pix = count_nonblack_np(output)
        ratio = color_pix / tot_pix
        # print("ratio is:", ratio)


        if ratio > 0.3 and i == 0:
            # print("white")
            # cv2.imshow("images", np.hstack([image, output]))
            # cv2.waitKey(0)
            c_green = True

        if ratio > 0.45 and ratio < 0.6 and i == 1:
            # print("green")
            # cv2.imshow("images", np.hstack([image, output]))
            # cv2.waitKey(0)
            green = ratio
            c_white = True

        i += 1

    if c_white == True and c_green == True:
        return 'ball', green


    else:
        return 'not_sure', green


def detect_team(image, show=False):
    # define the list of boundaries
    boundaries = [
        ([94, 80, 0], [126, 255, 255]),  # navy
        # ([57, 0, 170], [179,255, 255]) #cyan
        ([0, 50, 20], [5, 255, 255]),  # red
        ([0, 174, 200], [30, 255, 255]),  # orange
        ([0, 0, 0], [180, 255, 50]),  # black
        ([0, 0, 0], [179, 70, 255]),  # white
    ]

    i = 0
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)
        tot_pix = count_nonblack_np(image)
        color_pix = count_nonblack_np(output)
        ratio = color_pix / tot_pix
        #         print("ratio is:", ratio)
        if ratio > 0.01 and i == 0:
            return 'blue'
        elif ratio > 0.01 and i == 1:
            return 'red'
        elif ratio > 0.01 and i == 2:
            return 'orange'
        elif ratio > 0.01 and i == 3:
            return 'black'

        i += 1

        if show:
            cv2.imshow("images", np.hstack([image, output]))
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

    return 'not_sure'


@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])


# intializing the web camera device
# Be carefull with dimentions todo handle dynamic stuff
out = cv2.VideoWriter('soccer_out2.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (1920, 1080))

filename = 'soccer_arsenal.mp4'
cap = cv2.VideoCapture(filename)

# Running the tensorflow session
# with detection_graph.as_default():
# with tf.compat.v1.Session(graph=detection_graph) as sess:


counter = 0
pts = deque(maxlen=5)
ret, image_np = cap.read()

while ret:

    counter += 1

    h = image_np.shape[0]
    w = image_np.shape[1]

    if counter % 1 == 0:

        thresh_callback(image_np)

        image_np_expanded = np.expand_dims(image_np, axis=0)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()
        classe = []
        boxe = []
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = (detections['detection_classes'][0].numpy() + label_id_offset).astype(int)

        # boxes = np.flatten(boxes)
        # print ("boxes" , boxes)

        for n in range(len(classes)):
            if classes[n] == 1 or classes[n] == 37:
                classe.append(classes[n])

        for n in range(len(scores)):

            if scores[n] > 0.35:
                # Calculate position
                ymin = int(boxes[n][0] * h)
                xmin = int(boxes[n][1] * w)
                ymax = int(boxes[n][2] * h)
                xmax = int(boxes[n][3] * w)
                if xmax - xmin < 500:
                    boxe.append(boxes[n])

        # print("classe", classe)
        # print("classes", classes)
        # print ("boxes" , boxes)

        boxe = np.array(boxe)
        # print("boxe", boxe)


        image_np_with_detections = image_np

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            boxe,
            classe,
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.35,
            line_thickness=1,
            agnostic_mode=False)

        frame_number = counter
        loc = {}

        for n in range(len(scores)):

            label = None

            if scores[n] > 0.45:
                # Calculate position
                ymin = int(boxes[n][0] * h)
                xmin = int(boxes[n][1] * w)
                ymax = int(boxes[n][2] * h)
                xmax = int(boxes[n][3] * w)

                # Find label corresponding to that class
                for cat in categories:
                    if cat['id'] == classes[n]:
                        label = cat['name']

                ## extract every person
                if label == 'person':
                    # crop them
                    crop_img = image_np[ymin:ymax, xmin:xmax]
                    color = detect_team(crop_img)

                    if color != 'not_sure':

                        coords = (xmin, ymin)
                        if color == 'blue' or color == 'orange':
                            loc[coords] = 'LEI'
                        elif color == 'black':
                            loc[coords] = 'REF'
                        else:
                            loc[coords] = 'ARS'

        ## print color next to the person
        for key in loc.keys():
            text_pos = str(loc[key])
            cv2.putText(image_np, text_pos, (key[0], key[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0),
                        2)  # Text in black

    # cv2.imshow('image', image_np)
    # cv2.waitKey(0)
    out.write(image_np)
    # go to next image
    ret, image_np = cap.read()

    # if cv2.waitKey(0)  & 0xFF == ord('q'):
    #   cv2.destroyAllWindows()
    #   cap.release()
    #   break
print("done")
