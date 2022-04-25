import time

import cv2

import yolo_img

net, output_layers = yolo_img.load_yolo()
classes, colors_dict = yolo_img.class_color_dict("coco.names")

# Loading web cam
camera = cv2.VideoCapture(0)

text_display_count = 0

last = time.time()
while True:
    _, img = camera.read()

    class_ids, confidences, boxes = yolo_img.predict_yolo(img, net, output_layers)
    new_img = yolo_img.draw_yolo_predictions(img, classes, class_ids, confidences, boxes, colors_dict)

    height, width, _ = new_img.shape

    font = cv2.FONT_HERSHEY_PLAIN
    if text_display_count < 15:
        cv2.rectangle(new_img, (0, 0), (width, height), (0, 0, 255), 5)
        cv2.putText(new_img, "Intrus", (10, 50), font, 3, (0, 0, 255), 3)
    if text_display_count > 20:
        text_display_count = 0
    text_display_count += 1

    cv2.imshow("Image", new_img)

    current = time.time()
    print(round(1/(current-last), 1))
    last = current

    key = cv2.waitKey(1)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()