import cv2 as cv
import time
import numpy as np


COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

cap = cv.VideoCapture(0)

## Carregando os pesos da rede neural
net = cv.dnn.readNet("yolov4.weights", "yolov4.cfg")

## Settando os parâmetros da rede neural
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255)


## Leitura de Frames
while True:
    success, frame = cap.read()

    start_time = time.time()
    classes, scores, boxes = model.detect(frame, 0.1, 0.2)
    end_time = time.time()

    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = f"{class_names[classid]} : {score}"

        cv.rectangle(frame, box, color, 2)

        cv.putText(
            frame, label, (box[0], box[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    fps_label = f"FPS: {round((1.0/(end_time - start_time)),2)}"

    cv.putText(frame, fps_label, (0, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
    cv.putText(frame, fps_label, (0, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv.imshow("detection", frame)
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()
