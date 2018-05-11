import os
import cv2
import time
import requests
import subprocess

API_URL = "http://b10685df.ngrok.io/auth/"

print("Initializing the camera ...")
video_capture = cv2.VideoCapture(0)
# Warmup the camera
time.sleep(3)
for _ in range(50):
    video_capture.read()
print("Done initializing the camera ...")

# The average frame
avg = None

while True:
    """Main loop"""
    # Get the current frame, convert it to grayscale and blur it
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if avg is None:
        print("Initializing the average frame ...")
        avg = gray.copy().astype("float")
        # cv2.imshow("Avg", avg)
        # cv2.waitKey(0)
        continue

    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    thresh = cv2.threshold(frameDelta, 5, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        if cv2.contourArea(c) > 10000:
            subprocess.call(["./speak.sh", '"Hold still."'])
            for _ in range(50):
                video_capture.read()
            _, frame = video_capture.read()

            cv2.imwrite("tmp.jpg", frame)
            files = {'image': open("tmp.jpg", "rb")}
            resp = requests.post(API_URL, files=files)
            resp_json = resp.json()
            print(resp_json)
            if "error" not in resp_json:
                subprocess.call(["./speak.sh", u'"Hello, {}."'.format(resp_json["person"])])
                # subprocess.call(["./speak.sh", u'"Happy birthday to you, happy birthday to you, happy birthday dear {}, happy birthday to you.'.format(resp_json["person"])])
                subprocess.call(["./speak.sh", u'"{}"'.format(resp_json["weather"])])
                subprocess.call(["./speak.sh", u'"First let me tell you a joke: {}"'.format(resp_json["joke"])])
                subprocess.call(["./speak.sh", u'"{}"'.format(resp_json["feeds"])])
                subprocess.call(["./speak.sh", u'"{}"'.format(resp_json["todos"])])
            else:
                subprocess.call(["./speak.sh", '"No face detected."'])

            for _ in range(200):
                video_capture.read()

            break
