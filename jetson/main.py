import argparse
import os
import io
import time
import base64
from datetime import datetime
import threading
import signal

import socket

from firebase import firebase
from google.cloud import storage

from PIL import Image
import imutils
import cv2

import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from facenet_pytorch import MTCNN

import RPi.GPIO as GPIO

from model.triplet.model import TripletNet

device = "cpu"

ID = "hello"
detect_pin = 18
R_pin = 23
G_pin = 24
B_pin = 25

global_frame = None
running = True

img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((140, 140)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def exit_handler(signum, frame):
    global running
    running = False


def set_light(R_value, G_value, B_value):
    GPIO.output(R_pin, R_value)
    GPIO.output(G_pin, G_value)
    GPIO.output(B_pin, B_value)


def monitor():
    global global_frame
    global running

    BUFF_SIZE = 1024
    WIDTH = 200

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
    server_socket.bind(("0.0.0.0", 9999))
    server_socket.settimeout(5)

    while running:
        try:
            msg, client_addr = server_socket.recvfrom(BUFF_SIZE)
            print(client_addr)

            while running:
                curr_frame = global_frame.copy()
                frame = imutils.resize(curr_frame, width=WIDTH)
                encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                message = base64.b64encode(buffer)
                server_socket.sendto(message, client_addr)

                try:
                    msg = server_socket.recv(BUFF_SIZE)
                except socket.timeout as e:
                    print(e)
                    break

                time.sleep(0.01)
        except Exception as e:
            pass

    server_socket.close()


@torch.no_grad()
def detect_faces(mtcnn, bucket, database, model):
    global global_frame
    measure_dis = nn.PairwiseDistance(p=2)

    while running:
        value = GPIO.input(detect_pin)

        if value == GPIO.HIGH:
            set_light(0, 0, 1)

            now = datetime.now()
            curr_time = now.strftime("%Y-%m-%d %H:%M:%S")

            # Read in the frame
            frame = global_frame.copy()

            # Do the face alignment
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes, _ = mtcnn.detect(pil_img)

            if boxes is not None:
                boxes = boxes[0].astype("int").tolist()
                frame = frame[boxes[1]:boxes[3], boxes[0]:boxes[2]]

            # Detect the results
            if not (np.array(frame.shape) == 0).any():
                frame_tensor = img_transforms(frame).to(device)
                res = model.get_features(frame_tensor.unsqueeze(0)).cpu()

                name = "None"
                pass_status = False

                min_distance = 100
                # Face result
                identity = database.get(f"/users/{ID}", "identity")
                if identity is not None:
                    for key, value in identity.items():
                        # Get the value tensor
                        value_tensor = torch.tensor(value).to(device).unsqueeze(0)

                        # Compute the distance
                        distance = measure_dis(value_tensor, res).item()

                        if distance < 0.4:
                            if min_distance > distance:
                                name = key
                                pass_status = True
                                min_distance = distance

                    # Upload the data info
                    frame_upload(frame, curr_time, bucket)
                    upload_info(curr_time, database, res, pass_status, name)

                if pass_status:
                    set_light(0, 1, 0)
                    time.sleep(1)
                else:
                    set_light(1, 0, 0)
                    time.sleep(1)

                # Reset light
                set_light(0, 0, 0)

            # Reset light
            set_light(0, 0, 0)


def frame_upload(frame, filename, bucket):
    image_blob = bucket.blob(filename)
    temp_file = Image.fromarray(cv2.resize(frame, (480, 480)))
    temp_file_bytes = io.BytesIO()
    temp_file.save(temp_file_bytes, format="JPEG")

    # Read the bytes from beginning
    temp_file_bytes.seek(0)
    image_blob.upload_from_file(temp_file_bytes, content_type="image/jpeg")


def upload_info(curr_time, database, res, pass_status=False, name="None"):
    database.put(f"/users/{ID}/secure", curr_time, {
        "pass": pass_status,
        "name": name,
        "features": res.view(-1).tolist()
    })


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "-d", "--download", default=False,
        action="store_true"
    )
    args = parse.parse_args()

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(detect_pin, GPIO.IN)
    GPIO.setup(R_pin, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(G_pin, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(B_pin, GPIO.OUT, initial=GPIO.LOW)

    # Cool your device!
    set_light(1, 0, 0)
    time.sleep(0.2)
    set_light(0, 1, 0)
    time.sleep(0.2)
    set_light(0, 0, 1)
    time.sleep(0.2)
    set_light(0, 0, 0)

    # Set up google cloud client
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "face_identity.json"
    client = storage.Client()
    bucket = client.get_bucket("face_identity")

    # Set up camera
    cap = cv2.VideoCapture(0)
    time.sleep(1)

    # Set up firebase app
    database = firebase.FirebaseApplication(
        'https://face-identity-default-rtdb.asia-southeast1.firebasedatabase.app/', None
    )

    if args.download:
        # Download the pre-trained weight from google cloud
        print("Start downloading new weight ...")
        blob = bucket.blob("model.pt")
        blob.download_to_filename("weight/model.pt")
        print("Downloading end")

    # Set up the model
    model = TripletNet(pretrained=False).eval()
    model.load_state_dict(torch.load("weight/model.pt", map_location=device))

    # Face cropping
    mtcnn = MTCNN(select_largest=False, post_process=False, device=device)

    # Create a video thread for showing the video
    detect_faces_thread = threading.Thread(
        target=detect_faces,
        args=(mtcnn, bucket, database, model),
        daemon=True
    )
    detect_faces_thread.start()

    # Create a monitor thread
    monitor_thread = threading.Thread(
        target=monitor,
        daemon=True
    )
    monitor_thread.start()

    # Catch signal if necessary
    signal.signal(signal.SIGINT, exit_handler)

    # Upload information
    while running:
        _, frame = cap.read()
        frame = imutils.resize(frame, width=640)

        cv2.imshow("frame", frame)
        global_frame = frame
        cv2.waitKey(40)

    detect_faces_thread.join()
    monitor_thread.join()

    # Release the resources
    GPIO.cleanup()
    cap.release()
    print("\rProcess end")
