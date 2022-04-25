import time
from math import floor

import cv2

import yolo_img

# Load Yolo

net, output_layers = yolo_img.load_yolo()
classes, colors_dict = yolo_img.class_color_dict("coco.names")

vid_name = "cuisine"

camera = cv2.VideoCapture(vid_name + ".mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(vid_name + "_out.mp4", fourcc, 30, (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))))

n_frames = camera.get(cv2.CAP_PROP_FRAME_COUNT)

curr_frame = 1

start_time = time.time()
while True:
    retval, img = camera.read()
    if not retval:
        break
    class_ids, confidences, boxes = yolo_img.predict_yolo(img, net, output_layers)

    new_img = yolo_img.draw_yolo_predictions(img, classes, class_ids, confidences, boxes, colors_dict)

    writer.write(new_img)
    cv2.imshow("Image", new_img)
    cv2.waitKey(1)
    per_frame = (time.time()-start_time)/curr_frame
    remaining = (n_frames-curr_frame)*per_frame
    print(str(curr_frame) + "/" + str(round(n_frames)), str(round(curr_frame/n_frames*100)) + "%", " - Elapsed time:", str(floor(time.time()-start_time)) + "s", " - Remaining:", str(round(remaining/60, 2)) + " min")
    curr_frame += 1

camera.release()
writer.release()
cv2.destroyAllWindows()
