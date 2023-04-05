from tkinter import *
import cv2
from PIL import Image, ImageTk
from threading import Thread
import time
import argparse
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer
def server():
    global bear,cup
    class RequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                print("html")
                self.send_response(200, "OK")
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                document = "index.html"
                f = open(document)
                self.wfile.write(f.read().encode())
            elif "/data" in self.path:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                json_str = '{ "cup":'+str(cup)+', "bear":'+str(bear)+'}'
                self.wfile.write(json_str.encode(encoding='utf_8'))
                print("load")
            elif self.path == '/camera.png':
                self.send_response(200, "OK")
                self.send_header("Content-Type", "image/png")
                self.end_headers()

                self.wfile.write(open('camera.png', 'rb').read())
            else:
                self.send_response(404, "NOT FOUND")
    httpd = HTTPServer(('localhost', 8500), RequestHandler)
    httpd.serve_forever()
def open_camera():
    global frame,frame_out,width
    online_true, frame = vid.read()
    frame_in_local=frame
    frame_out_local=frame_out
    frame_out_local=cv2.resize(frame_out_local, (512, 288))
    frame_in_local=cv2.resize(frame_in_local, (512, 288))
    opencv_image = cv2.cvtColor(frame_out_local, cv2.COLOR_BGR2RGBA)
    opencv_image2 = cv2.cvtColor(frame_in_local, cv2.COLOR_BGR2RGBA)
    captured_image = Image.fromarray(opencv_image)
    captured_image2 = Image.fromarray(opencv_image2)
    photo_image = ImageTk.PhotoImage(image=captured_image)
    photo_image2 = ImageTk.PhotoImage(image=captured_image2)
    label_widget.photo_image = photo_image
    label_widget.configure(image=photo_image)
    label_widget.place(x= 0, y = 0)
    label_widget.after(10, open_camera)
    label_widget2.photo_image = photo_image2
    label_widget2.configure(image=photo_image2)
    label_widget2.place(x= 512, y = 0)
def get_output_layers(net):
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers
def recognise():
    def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])
        color = COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    global frame,frame_out,bear,cup
    print("start load model")
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    args.image = '1.jpeg'
    args.config = 'yolov3.cfg'
    args.weights = 'yolov3.weights'
    args.classes = 'yolov3.txt'
    scale = 0.00392
    net = cv2.dnn.readNet(args.weights, args.config)

    while True:
        image = frame
        print("read frame")
        Width = image.shape[1]
        Height = image.shape[0]

        classes = None
        with open(args.classes, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        COLORS = np.random.uniform(150, 255, size=(len(classes), 3))
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        bear = 0
        cup = 0
        for i in indices:
            try:
                box = boxes[i]
            except:
                i = i[0]
                box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            if(class_ids[i]==77):
                bear=1
            if(class_ids[i]==41):
                cup=1
            if((class_ids[i]==77)or(class_ids[i]==41)):
                draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
        frame_out=image
        print("do recognise")
        time.sleep(0.2)
print("start")
frame_out= cv2.imread('camera.png')
vid = cv2.VideoCapture(1)
cup=0;
bear=0;
print("connect to USB-camera")
app = Tk()
app.geometry("1024x288")
app.bind('<Escape>', lambda e: app.quit())
label_widget = Label(app)
label_widget2 = Label(app)
label_widget.pack()
label_widget2.pack()
width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
thread_cam = Thread(target=open_camera, args=())
thread_cam.start()
thread_rec = Thread(target=recognise, args=())
thread_rec.start()
thread_server = Thread(target=server, args=())
thread_server.start()
app.mainloop()