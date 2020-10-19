# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from imutils.video import FPS
import os

#videoPath = "test.mp4"

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--prototxt", required=True,
  #              help="path to Caffe 'deploy' prototxt file")
#ap.add_argument("-m", "--model", required=True,
      #          help="path to Caffe pre-trained model")

ap.add_argument("-i", "--input", type=str, default="",
                help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
                help="path to (optional) output video file")
ap.add_argument("-c", "--confidence", type=float, default=0.7,
                help="minimum probability to filter weak detections")
ap.add_argument("-m", "--mask-rcnn", required=True,
                help="base path to mask-rcnn directory")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="minimum threshold for pixel-wise mask segmentation")
args = vars(ap.parse_args())

##############################################################
# load the COCO class labels our Mask R-CNN was trained on
labelsPath = os.path.sep.join([args["mask_rcnn"],
                               "object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],
                                "frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
                               "mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# load our Mask R-CNN trained on the COCO dataset (90 classes)
# from disk
print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)
##############################################################

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load our serialized model from disk
# print("[INFO] loading model...")
# net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
# vs = VideoStream(videoPath).start()
vs = cv2.VideoCapture(args["input"])
writer = None
#time.sleep(2.0)

########################
#writer = None
#fps = FPS().start()
########################

# loop over the frames from the video stream
while True:
    # read the next frame from the video stream and resize it
    # frame = vs.read()
    grabbed, frame = vs.read()

    #frame = imutils.resize(frame, width=400)

    # if the frame dimensions are None, grab them
    # if W is None or H is None:
    (H, W) = frame.shape[:2]

    # construct a blob from the frame, pass it through the network,
    # obtain our output predictions, and initialize the list of
    # bounding box rectangles
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (104.0, 177.0, 123.0))
    # net.setInput(blob)
    # detections = net.forward()
    # rects = []

    #blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(["detection_out_final", "detection_masks"])
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
    rects = []  # for centroid
    # loop over the detections
    for i in range(0, boxes.shape[2]):
        # filter out weak detections by ensuring the predicted
        # probability is greater than a minimum threshold

        classID = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object, then update the bounding box rectangles list
            # (H, W) = frame.shape[:2]
            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))  # for centroid

            # draw a bounding box surrounding the object so we can
            # visualize it
            (startX, startY, endX, endY) = box.astype("int")
            # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            boxW = endX - startX
            boxH = endY - startY
            # extract the pixel-wise segmentation for the object,
            # resize the mask such that it's the same dimensions of
            # the bounding box, and then finally threshold to create
            # a *binary* mask
            mask = masks[i, classID]
            mask = cv2.resize(mask, (boxW, boxH),
                              interpolation=cv2.INTER_CUBIC)
            mask = (mask > args["threshold"])

            # extract the ROI of the image but *only* extracted the
            # masked region of the ROI
            roi = frame[startY:endY, startX:endX][mask]

            # grab the color used to visualize this particular class,
            # then create a transparent overlay by blending the color
            # with the ROI
            color = COLORS[classID]
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

            # store the blended ROI in the original frame
            frame[startY:endY, startX:endX][mask] = blended

            # draw the bounding box of the instance on the frame
            color = [int(c) for c in color]
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          color, 2)

            # draw the predicted label and associated probability of
            # the instance segmentation on the frame
            text = "{}: {:.4f}".format(LABELS[classID], confidence)
            cv2.putText(frame, text, (startX, startY - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    if args["output"] != "" and writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

    # if the video writer is not None, write the frame to the output
    # video file
    if writer is not None:
        writer.write(frame)

vs.stop()
# do a bit of cleanup
cv2.destroyAllWindows()

