import requests

import cv2
import numpy as np
import glob
import os
import threading
from PIL import Image

from natsort import natsorted

import time as pytime
import datetime

##### Connect to databse for using the existing embeddings for comparison #####
import pymongo

# from ssh_pymongo import MongoSession

# session = MongoSession('128.199.28.197',
#     port=22,
#     user='digiadmin',
#     password='D!g!t3Le@123',
#     uri='mongodb://root:y62BXPqjbUAa2LDRBp3Gxu2bbgDFCNxX@128.199.28.197/'
# )

# db = session.connection['vms']
connection = pymongo.MongoClient('mongodb://root:xYrnn9CK4wyJLdWDm9L5kGGc@10.83.0.160:27017/')

db = connection.Face_Recognition_Database

date = datetime.date.today()
todays_data = str(date)

coll = db['fr_results']

#coll.create_index([('date', pymongo.ASCENDING)], unique=True)

#################################################
infeed_path = '/DATA02/ml_Workspaces/FACE_RECOGNITION/in_feed_'
#path_of_test_images = infeed_path + todays_data + "/"
path_of_test_images = infeed_path + "/"

# path_of_test_images = "/DATA02/ml_Workspaces/FACE_RECOGNITION/Test_Images"
# print('path_of_test_images = ',path_of_test_images)

color1 = (0, 255, 0)
color2 = (0, 0, 255)
font = cv2.FONT_HERSHEY_PLAIN
count = 1

frames_write_path = '/DATA02/ml_Workspaces/FACE_RECOGNITION/Processed_Images_entry_'+'/'
results_path = '/DATA02/ml_Workspaces/FACE_RECOGNITION/FR_entry_results_' + '/'

# frames_write_path = 'D:/Pycharm projects/A_FR/'+todays_data+'_Processed_Images_entry/'
# results_path = 'D:/Pycharm projects/A_FR/'+todays_data+'_FR_entry_results/'

if os.path.exists(frames_write_path) is False:
    os.mkdir(frames_write_path)

if os.path.exists(results_path) is False:
    os.mkdir(results_path)

print('......Now Start Camera Capture.....')

class Async_Process(threading.Thread):

    def __init__(self, frame, port):
        threading.Thread.__init__(self)
        self.frame = frame
        self.Result = None
        self.face = None
        self.bbox = None
        self.port = port

    def run(self):
        params = {}
        data = {'params': params, 'image': self.frame}

        resp1 = requests.post("http://10.83.0.201:" + self.port + "/get_face_recognition", json=data)

        out_dict = resp1.json()
        # print('out_dict : ', out_dict)

        self.face, self.Result, self.bbox = list(out_dict.values())[0]
        print('Length = ',len(list(out_dict.values())[0]))
        print('Result in process : ',self.Result)
        print('bbox in process : ',self.bbox)


    def pred(self):
        return self.face, self.Result, self.bbox


class AsyncWrite(threading.Thread):
    def __init__(self, filename, frame):
        threading.Thread.__init__(self)
        self.frame = frame
        self.filename = filename

    def run(self):
        cv2.imwrite(self.filename, self.frame)


class AsyncWriteDB(threading.Thread):

    def __init__(self, plbl, face_img, intime,isodate):
        threading.Thread.__init__(self)
        self.plbl = plbl
        self.face_img = face_img
        self.intime = intime
        self.isodate = isodate

    def run(self):
       e_name, e_id = self.plbl.split('_')
       # e_id = self.plbl.split('_')[1]
       face_img = cv2.cvtColor(self.face_img, cv2.COLOR_RGB2BGR)
       insert = 'True'

       date = datetime.date.today()
       todays_data = str(date)
       print('todays_data ##########',todays_data)


       for doc in coll.find({"employeeid": e_id, 'date': todays_data}):
           insert = 'False'
       
       if insert == 'True':
           wr = cv2.imwrite(results_path + self.plbl + '_' + todays_data.replace(':', '-') + '_' + self.intime.replace(':', '-') + '_' + '.jpg', face_img)
           try:
               face_imgdb = face_img.tolist()
               s_no = coll.count_documents({}) + 1
               coll.insert_one(
                   {'serialno': s_no, 'employeename': e_name, 'employeeid': e_id, 'createdon': self.isodate,
                    'date': todays_data, 'intime': self.intime, 'outtime': '-',
                    'totalworkhours': '-', 'image': ''})
               print('##  Inserted into db @ at time : ', datetime.datetime.now())
               inserted = 'True'
               # pytime.sleep(1)

           except:
               inserted = 'Already Exists'
               print('Unable to insert')
               pass

class AsyncDrawFrames(threading.Thread):
    def __init__(self, framecv, Results, bboxes):
        threading.Thread.__init__(self)
        self.framecv = framecv
        self.Results = Results
        self.bboxes = bboxes

    def run(self):
        for fri, frl in enumerate(self.framecv):
            frame_cv = np.array(frl, dtype=np.uint8)
            if self.bboxes[fri] is not None:
                x, y, w, h = self.bboxes[fri]
                text = self.Results[fri]
                frame_cv = cv2.rectangle(frame_cv, (x, y), (w, h), color1, 2)
                frame_cv = cv2.putText(frame_cv, text, (20, 20), font, 1, color2, 2)
            else:
                text = self.Results[fri]
                frame_cv = cv2.putText(frame_cv, text, (20, 10), font, 1, color2, 2)
            frame_cv = cv2.cvtColor(frame_cv, cv2.COLOR_RGB2BGR)
            t = str(datetime.datetime.now()).replace(':', '-')
            # background = AsyncWrite(frames_write_path + t + '_.jpg', frame_cv)
            # background.start()
            cv2.imwrite(frames_write_path + t + '_.jpg', frame_cv)
            pytime.sleep(0.5)


class AsyncDelete(threading.Thread):
    def __init__(self, path):
        threading.Thread.__init__(self)
        self.path = path

    def run(self):
        os.remove(self.path)
        pytime.sleep(2)


results_list = []
plbl = None
face_img = None
f_c = 0
files_list = []
Result = None

frames_list = []
while True:
    files = natsorted(glob.glob(path_of_test_images + '/*.jpg'))
    if len(files) <= 1:
        #print('No images in the given path....')
        # break
        # pytime.sleep(3)
        continue
    # elif len(files) == 2:
    # pytime.sleep(1)
    #  continue
    # frames_list = []
    detected_faces_list = []
    print('file 0 before read', files[0], '....time..', datetime.datetime.now())
    try:
        frame_cv1 = cv2.imread(files[0])
        #frame_cv1 = cv2.resize(frame_cv1, (640,480), interpolation = cv2.INTER_AREA)
    except:
        frame_cv1 = None
    print('file 0 after read', files[0], '....time..', datetime.datetime.now())
    if frame_cv1 is not None:
        # frame_cv = frame_cv1[0:1080, 700:1500]
        #frame_cv = frame_cv1[0:480, 200:550]
        frame_cv = frame_cv1[0:480, 60:580]
        frameRGB = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB).tolist()
        frames_list.append(frameRGB)

        # print('starting process')
        process1 = Async_Process(frames_list[0], port="4103")
        process1.start()

        print('.start delete ##', datetime.datetime.now())
        delete = AsyncDelete(files[0])
        delete.start()

    try:
        if files[1] is not None:
            print('file 1 before read', files[1], '....time..', datetime.datetime.now())
        frame_cv2 = cv2.imread(files[1])
        #frame_cv2 = cv2.resize(frame_cv2, (640, 480), interpolation=cv2.INTER_AREA)
    except:
        frame_cv2 = None

    if frame_cv2 is not None:
        print('file 1 after read', files[1], '....time..', datetime.datetime.now())
        # frame_cv = frame_cv2[0:1080, 700:1500]
        #frame_cv = frame_cv2[0:480, 200:550]
        frame_cv = frame_cv2[0:480, 60:580]
        frameRGB = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB).tolist()
        frames_list.append(frameRGB)

        # print('running')
        if len(frames_list) == 1:
            continue
            #process2 = Async_Process(frames_list[0], port="4104")
        else:
            process2 = Async_Process(frames_list[1], port="4104")
        process2.start()

    Results = [None, None]
    bboxes = [None, None]
    faces_list = [None, None]

    if frame_cv1 is not None:
        print('file 0 before join ', files[0])
        print('process 1 join ;  ', datetime.datetime.now())
        process1.join()

        print('after p1 join #########', datetime.datetime.now())

        faces_list[0], Results[0], bboxes[0] = process1.pred()

        if Results[0] is not None:
            # for ri, res in enumerate(Results[0]):
            res = Results[0]
            draw_res = res
            print('Frame Result = ', res)
            res = res.split('-')[0]
            if plbl == res:
                pass
            elif res == 'DETECTIONS = NONE':
                # draw = AsyncDrawFrames([frames_list[0]], [res], [None])
                # draw.start()
                pass
            elif res == 'Please move closer to the camera':
                #draw = AsyncDrawFrames([frames_list[0]], [res], [None])
                #draw.start()
                pass
            elif res == 'Unidentified':
                print('Unidentified', bboxes[1])
                #draw = AsyncDrawFrames([frames_list[0]], [draw_res], [bboxes[0]])
                #draw.start()
                pass
            elif res != 'Unidentified':
                plbl = res
                # print('plbl = ',plbl)
                # intime = pytime.strftime("%H:%M")
                isodate = datetime.datetime.now()  # .strftime('%Y-%m-%d %H:%M')
                intime = isodate.time().strftime('%H:%M')

                face_img = np.array(faces_list[0], dtype=np.uint8)

                print('############### Before Inserting in DB  p1   ############', datetime.datetime.now())
                background = AsyncWriteDB(plbl, face_img, intime,isodate)
                background.start()

                draw = AsyncDrawFrames([frames_list[0]], [draw_res], [bboxes[0]])
                draw.start()



    if frame_cv2 is not None:
        delete = AsyncDelete(files[1])
        delete.start()
        print('file 1 ', files[1])
        print('process 2 join ;', datetime.datetime.now())
        process2.join()

        print('after p2 join #########', datetime.datetime.now())

        faces_list[1], Results[1], bboxes[1] = process2.pred()

        if Results[1] is not None:
            # for ri, res in enumerate(Results[0]):
            res = Results[1]
            draw_res = res
            print('Frame Result = ', res)
            res = res.split('-')[0]
            if plbl == res:
                pass
            elif res == 'DETECTIONS = NONE':
                # draw = AsyncDrawFrames([frames_list[1]], [res], [None])
                # draw.start()
                pass
            elif res == 'Please move closer to the camera':
                #draw = AsyncDrawFrames([frames_list[1]], [res], [None])
                #draw.start()
                pass
            elif res == 'Unidentified':
                print('Unidentified',bboxes[1])
                #draw = AsyncDrawFrames([frames_list[1]], [draw_res], [bboxes[1]])
                #draw.start()
                pass
            elif res != 'Unidentified':
                plbl = res
                # print('plbl = ',plbl)
                isodate = datetime.datetime.now()  # .strftime('%Y-%m-%d %H:%M')
                intime = isodate.time().strftime('%H:%M')

                face_img = np.array(faces_list[1], dtype=np.uint8)

                print('############### Before Inserting in DB  p1   ############', datetime.datetime.now())
                background = AsyncWriteDB(plbl, face_img, intime,isodate)
                background.start()

                draw = AsyncDrawFrames([frames_list[1]], [draw_res], [bboxes[1]])
                draw.start()


    frames_list = []
