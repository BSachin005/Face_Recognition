##opencv - 4.1.2.30
import datetime
from PIL import Image
from math import sqrt, pow
import numpy as np


import torch
import torchvision.transforms as transforms

##### Connect to databse for using the existing embeddings for comparison #####
import pymongo

# from ssh_pymongo import MongoSession
#
# session = MongoSession('128.199.28.197',
#     port=22,
#     user='digiadmin',
#     password='D!g!t3Le@123',
#     uri='mongodb://root:y62BXPqjbUAa2LDRBp3Gxu2bbgDFCNxX@128.199.28.197/'
# )
#
# db = session.connection['Face_Recognition_Database']
connection = pymongo.MongoClient('mongodb://root:xYrnn9CK4wyJLdWDm9L5kGGc@10.83.0.160:27017/')

db = connection.Face_Recognition_Database

# date = datetime.date.today()
# todays_data = str(date)
#
# coll = db[todays_data + '_FR_results']
#
# coll.create_index([('employee_id', pymongo.ASCENDING), ('in_time', pymongo.ASCENDING)], unique=True)

from flask import Flask, jsonify, request



from facenet_pytorch import MTCNN,InceptionResnetV1, fixed_image_standardization

standard_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

device = 'cpu'

model = InceptionResnetV1(pretrained='vggface2', device=device).eval()

print('Model loaded..........API 4105')

def embedding(t):
    embed = model(t.to(device))
    embed = embed.data
    return embed

face_detector = MTCNN(min_face_size=50,thresholds=[0.6,0.7,0.8,0.9])

def face_direction(face_keypoints):
    # x,y,w,h = bounding_box

    # center = (int((x+x+w)/2),int((y+y+h)/2))
    # print("Center = ",center)

    left_eye , right_eye , nose , mouth_left , mouth_right   = face_keypoints

    c, d = nose

    xe1, ye1 = left_eye
    xe2, ye2 = right_eye

    # xm1, ym1 = mouth_left
    # xm2, ym2 = mouth_right

    dist_RN = sqrt(pow(xe1 - c, 2) + pow(ye1 - d, 2) * 1.0)
    # print('Distance between right eye and nose = ',dist_RN)

    dist_LN = sqrt(pow(xe2 - c, 2) + pow(ye2 - d, 2) * 1.0)
    # print('Distance between left eye and nose = ',dist_LN)

    # deyes = dist_RN - dist_LN

    dr = dist_RN / dist_LN

    direction = ''

    if 0.8 < dr < 1.2:
        direction = 'straight'

    if 0.6 < dr < 0.8:
        direction = 'slight_right'

    if 1.2 < dr < 1.5:
        direction = 'slight_left'

    if dr < 0.6:
        direction = 'right'

    if dr > 1.5:
        direction = 'left'

    return direction


def face_detection(pil_img):
    # directions_list , bounding_boxes_list, cropped_faces_list = [], [], []

    bboxes, probs, keypoints = face_detector.detect(pil_img, landmarks=True)

    if bboxes is None:
       return None,None,None

    bounding_boxes = bboxes.tolist()

    facial_landmarks = keypoints.tolist()

    ### Taking only the first detected face which is the largest face #######
    # for i, bb in enumerate(bounding_boxes):
    x, y, w, h = bounding_boxes[0]
    # if x < 100:
    #     continue
    cropped_img = pil_img.crop((x, y, w, h))

    im1_w, im1_h = cropped_img.size[:2]
    # print('Size of test face = ',im1_w,im1_h)

    if im1_w < 35 or im1_h < 35:
        # text = 'Please move closer to the camera'
        text = 'closer'
        return text, None, None

    face_keypoints = facial_landmarks[0]
    direction = face_direction(face_keypoints)

    # cropped_faces_list.append(cropped_img)
    # directions_list.append(direction)
    bbox = list(map(int, bounding_boxes[0]))
    # bounding_boxes_list.append(bbox)
    # bounding_boxes_list = [list(map(int, bbl)) for bbl in bounding_boxes_list]
    return cropped_img, direction, bbox


def face_recognition(img_test,im_d):
  direction = 'straight' #im_d
  print('direction',direction)
  collection = db[direction]
  # print('collection = ',collection)
  check = [i['face_embeddings'] for i in collection.find({},{'face_embeddings':1})]

  im1g = img_test
  im1gt = standard_transform(im1g)
  x0 = im1gt.unsqueeze(0)
  output1 = embedding(x0)

  outputs2_list = [torch.tensor(c).to(device) for c in check]

  euclidean_distance = [torch.cdist(output1, output2).min().item() for output2 in outputs2_list]
  # print('Euclidean distance = ',euclidean_distance)
  min_dist = min(euclidean_distance)
  print('distnace score min_dist = ',min_dist)
  if min_dist>0.66:
      return 'Unidentified-' + str(min_dist)  # ...Please provide a better image and try again' #'Person Unidentified'
  else:
      indx = euclidean_distance.index(min_dist)
      #print('Index = ',indx)
      c = check[indx]
      myquery = { "face_embeddings": c }
      mydoc = collection.find(myquery)
      label_name = mydoc[0]['name']
      emp_id = mydoc[0]['employee_id']
      return label_name + '_' + emp_id +'-'+str(min_dist)  # 'Person Identified as '+label_name

def final(pil_img):
    cropped_img, direction, bbox = face_detection(pil_img)
    print('Face Detection is done')
    if cropped_img is None:
        Result = 'DETECTIONS = NONE'
    elif cropped_img == 'closer':
        Result = 'Please move closer to the camera'
        bbox = None
    else:
        Result = face_recognition(cropped_img, direction)
    cropped_img = np.array(cropped_img).tolist()
    Results_list = [cropped_img, Result, bbox]
    print('Face Recognition is done')
    res = {'result': Results_list}
    return res

app = Flask(__name__)

@app.route('/get_face_recognition', methods=['POST'])
def get_face_recognition():
    if request.method == 'POST':
        image = np.array(request.json['image'], dtype=np.uint8)
        # print('image shape for face detection = ',image.shape)
        pil_img = Image.fromarray(image)
        res = final(pil_img)
        return jsonify(res)


host = '10.83.0.201'
port = 4105


# if __name__ == '__main__':
app.run(host=host,port=port,debug=True)
