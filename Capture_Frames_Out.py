import os

import cv2
import threading
import datetime
import time as pytime


path_to_store_images = '/DATA02/ml_Workspaces/FACE_RECOGNITION/out_feed_'

# path_to_store_images = 'D:/Pycharm projects/Current_FR/New_Capture_for_Videos/'
#
date = datetime.date.today()
#todays_data_path = path_to_store_images + str(date) + '/'
todays_data_path = path_to_store_images + '/'

if os.path.exists(todays_data_path) is False:
    os.mkdir(todays_data_path)

class AsyncWrite(threading.Thread):

    def __init__(self, filename, frame):
        threading.Thread.__init__(self)
        self.frame = frame
        self.filename = filename

    def run(self):
        cv2.imwrite(self.filename, frame)
        # time.sleep(2)
    # print("Finished background file write to",self.filename)

pytime.sleep(5)
print('Running')

# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('rtsp:/digitele:drone@123@10.84.1.91/video')     ###in camera
cap = cv2.VideoCapture('rtsp:/digitele:drone@123@10.84.1.92/video')   ###out camera

count = 1
fr_rate = 0
restarted = 0
while True:
    # time.sleep(1)
    status, frame = cap.read()
    if status is False:
        print('camera feed unavailable...restarting')
        #cap = cv2.VideoCapture('rtsp:/digitele:drone@123@10.84.1.91/video')     ###in camera
        cap = cv2.VideoCapture('rtsp:/digitele:drone@123@10.84.1.92/video')   ###out camera
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        restarted = 1
        continue

    if restarted == 1:
        print('Restarted and Running')
        restarted = 0

    #
    fr_rate += 1
    if fr_rate % 25 != 0:
        continue

    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)

    background = AsyncWrite(todays_data_path + str(datetime.datetime.now()).replace(':','-') + '.jpg', frame)
    background.start()
    count+=1