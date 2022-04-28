import time
import cv2
import yolo_img

net, output_layers = yolo_img.load_yolo()
classes, colors_dict = yolo_img.class_color_dict("coco.names")

# Loading web cam
camera = cv2.VideoCapture(0)

last = time.time()
while True:
    _, img = camera.read()

    class_ids, confidences, boxes = yolo_img.predict_yolo(img, net, output_layers)
    new_img = yolo_img.draw_yolo_predictions(img, classes, class_ids, confidences, boxes, colors_dict)

    height, width, _ = new_img.shape

    cv2.imshow("Image", new_img)

    current = time.time()
    print(round(1/(current-last), 1))
    last = current

    key = cv2.waitKey(1)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()