import cv2

import yolo_img

net, output_layers = yolo_img.load_yolo()
classes, colors_dict = yolo_img.class_color_dict("coco.names")

vid_name = "intrus_salon"
camera = cv2.VideoCapture(vid_name + ".mp4")

n_frames = camera.get(cv2.CAP_PROP_FRAME_COUNT)
fps = camera.get(cv2.CAP_PROP_FPS)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

def init_video_writer(filename):
    # Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filename + ".mp4", fourcc, fps, (width, height))
    return writer

text_display_count = 0
def draw_intruder(img):
    global text_display_count
    font = cv2.FONT_HERSHEY_PLAIN
    if text_display_count < 15:
        cv2.rectangle(img, (0, 0), (width, height), (0, 0, 255), 5)
        cv2.putText(img, "Intrus", (10, 50), font, 3, (0, 0, 255), 3)
    if text_display_count > 20:
        text_display_count = 0
    text_display_count += 1

def reset_draw_intruder():
    global text_display_count
    text_display_count = 0


out_index = 0
writer = init_video_writer(vid_name + "_intrus_" + str(out_index))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
global_writer = cv2.VideoWriter(vid_name + "_out" + ".mp4", fourcc, fps, (width, height))

intruder = False
recording = False
frame_noone = -1

while True:
    retval, img = camera.read()
    if not retval:
        break

    # Person detection
    class_ids, confidences, boxes = yolo_img.predict_yolo(img, net, output_layers)
    intruder = False
    for class_id in class_ids:
        label = classes[class_id]
        if label == "person":
            intruder = True
            frame_noone = -1
            break

    new_img = yolo_img.draw_yolo_predictions(img, classes, class_ids, confidences, boxes, colors_dict)

    if intruder:
        recording = True
        draw_intruder(new_img)
        writer.write(new_img)
    else:
        if recording:
            if frame_noone == -1:
                frame_noone = camera.get(cv2.CAP_PROP_POS_FRAMES)
            else:
                curr_frame = camera.get(cv2.CAP_PROP_POS_FRAMES)
                draw_intruder(new_img)
                writer.write(new_img)
                if (curr_frame - frame_noone) / fps > 10:
                    writer.release()
                    out_index += 1
                    reset_draw_intruder()
                    writer = init_video_writer(vid_name + "_intrus_" + str(out_index))
                    frame_noone = -1
                    recording = False
    global_writer.write(new_img)

    cv2.imshow("Detection d'intrus", new_img)
    cv2.waitKey(1)

writer.release()
global_writer.release()
camera.release()
cv2.destroyAllWindows()
