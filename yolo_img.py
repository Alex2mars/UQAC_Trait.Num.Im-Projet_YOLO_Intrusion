import cv2
import numpy as np

def load_yolo(weights_filename="yolov3.weights", config_filename="yolov3.cfg"):
    net = cv2.dnn.readNet(weights_filename, config_filename)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

def class_color_dict(class_filename):
    with open(class_filename, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    colors_dict = {cl: col for (cl, col) in zip(classes, colors)}
    return classes, colors_dict

def predict_yolo(img, yolo_net, output_layers):
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (220, 220), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    class_ids = [class_ids[i] for i in indexes]
    confidences = [confidences[i] for i in indexes]
    boxes = [boxes[i] for i in indexes]

    return class_ids, confidences, boxes

def draw_yolo_predictions(img, classes, class_ids, confidences, boxes, colors_dict):
    img = img.copy()
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors_dict[label]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + str(int(confidences[i]*100)) + "%", (x, y + 30), font, 2, color, 2)
    return img
