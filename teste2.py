import cv2 as cv
import numpy as np


COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

cap = cv.VideoCapture(0)
wth = 320
confThreshold = 0.5
nmsThreshold = 0.3

## Carregando a rede
modelCfg = "yolov3.cfg"
modelWeights = "yolov3.weights"
net = cv.dnn.readNet(modelCfg, modelWeights)

## Settando os parâmetros da rede neural
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

""" model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255)
 """


def findObjects(outputs, frame):
    ht, wt, ct = frame.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for detc in output:
            scores = detc[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            print(confidence)
        if confidence > confThreshold:
            w, h = int(detc[2] * wt), int(detc[3] * ht)
            x, y = int((detc[0] * wt) - w / 2), int((detc[1] * ht) - h / 2)

            bbox.append([x, y, w, h])
            classIds.append(classId)
            confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[3]
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv.putText(
            frame,
            f"{class_names[classId[i]].upper()} {int(confs[i]*100)}% ",
            (x, y - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
        )


## Leitura de Frames
while True:
    success, frame = cap.read()

    blob = cv.dnn.blobFromImage(frame, 1 / 255, (wth, wth), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()

    outputLayers = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputLayers)

    findObjects(outputs, frame)

    cv.imshow("Image", frame)

    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
