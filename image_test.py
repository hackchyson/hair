"""
单个图片检测任务
"""
from MtcnnDetector import MtcnnDetector
from detector import Detector
from fcn_detector import FcnDetector
from mtcnn_model import P_Net, R_Net, O_Net
from loader import TestLoader
import cv2
import os
import json
import numpy as np

json_config = "config/hair.json"
with open(json_config, "r") as json_file:
    data = json.load(json_file)

    x_shift_ratio = data['x_shift_ratio']
    y_shift_ratio = data['y_shift_ratio']
    x_widen = data["x_widen"]
    y_widen = data['y_widen']

test_mode = "ONet"
thresh = [0.9, 0.6, 0.7]
# thresh = [0.8, 0.7, 0.6]

min_face_size = 24
stride = 2
slide_window = False
shuffle = False
detectors = [None, None, None]
prefix = ['model/MTCNN_model/PNet_landmark/PNet', 'model/MTCNN_model/RNet_landmark/RNet',
          'model/MTCNN_model/ONet_landmark/ONet']
epoch = [18, 14, 16]
batch_size = [2048, 256, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
print(model_path)
# load pnet model
if slide_window:
    PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
else:
    PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet

# load rnet model
if test_mode in ["RNet", "ONet"]:
    RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
    detectors[1] = RNet

# load onet model
if test_mode == "ONet":
    ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
    detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)
gt_imdb = []
# gt_imdb.append("35_Basketball_Basketball_35_515.jpg")
# imdb_ = dict()"
# imdb_['image'] = im_path
# imdb_['label'] = 5


path = "data"

# hack
path = "data/source"

for item in os.listdir(path):
    if ('jpg' not in item):
        continue
    gt_imdb.append(os.path.join(path, item))

print(gt_imdb)

# gt_imdb = ['data/source/Zhang_Ziyi_0001.jpg']

test_data = TestLoader(gt_imdb)
all_boxes, landmarks = mtcnn_detector.detect_face(test_data)
print('box num', len(all_boxes[0]))
count = 0

for imagepath in gt_imdb:
    image = cv2.imread(imagepath)

    hair = cv2.imread('data/hairs/1111.png')
    # rows, cols, channels = hair.shape

    # print(hair.shape, image.shape)
    # image[0:rols, 0:cols] = hair

    for bbox, landmark in zip(all_boxes[count], landmarks[count]):
        cv2.putText(image, str(int(bbox[0])) + ',' + str(int(bbox[1])), (0, 100),
                    cv2.FONT_HERSHEY_TRIPLEX, 1,
                    color=(255, 0, 255))

        print(bbox)

        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)
        # ####################### 1
        # #
        # hair = cv2.resize(hair, (int((bbox[2] - bbox[0])), int(bbox[3] - bbox[1])))
        # rols, cols, channels = hair.shape
        # o_x = int(bbox[0] - (bbox[2] - bbox[0]) * 4 / 10)
        # o_y = int(bbox[1])
        # # print(o_x, o_y)
        # # image[o_x:rols + o_x, o_y: cols + o_y] = hair
        #
        # ######################### 2
        # roi = cv2.addWeighted(hair, 0.8, image[o_x:rols + o_x, o_y: cols + o_y], 0.4, 0.0)
        # image[o_x:rols + o_x, o_y: cols + o_y] = roi

        # for landmark in landmarks[count]:
        #     for i in range(int(len(landmark)/2)):
        #         cv2.circle(image, (int(landmark[2*i]),int(int(landmark[2*i+1]))), 3, (0,0,255))

        ############################ 3
        hair = cv2.resize(hair, (int((bbox[2] - bbox[0]) * x_widen), int((bbox[3] - bbox[1]) * y_widen)))
        rows, cols, channels = hair.shape

        o_x = int(bbox[1] - (bbox[3] - bbox[1]) * x_shift_ratio)
        o_y = int(bbox[0] - (bbox[2] - bbox[0]) * y_shift_ratio)

        cv2.putText(image, str(int(o_x)) + ',' + str(int(o_y)), (0, 50),
                    cv2.FONT_HERSHEY_TRIPLEX, 1,
                    color=(255, 0, 255))
        if o_x < 0 or o_y < 0:
            print("o_x: {}; o_y: {}".format(o_x, o_y))
            continue
        # o_x, o_y = 0,50

        cv2.rectangle(image, (o_x, o_y), (o_x + cols, o_y + rows), (0, 0, 255), 1)

        # Now create a mask of logo and create its inverse mask also
        img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 这个254很重要
        mask_inv = cv2.bitwise_not(mask, dst=None, mask=None)
        #
        # cv2.imshow('mask', mask_inv)
        # Now black-out the area of logo in ROI
        print(rows, cols, channels)
        #
        roi = image[o_x:o_x + rows, o_y:o_y + cols]

        print("o_x: {}; o_y: {}".format(o_x, o_y))
        print(image.shape)
        print(hair.shape)
        print('box: ', bbox)
        print('roi', roi.shape)
        #
        img2_fg = cv2.bitwise_and(hair, hair, mask=mask_inv)  # 这里才是mask_inv
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask)

        #
        # # Take only region of logo from logo image.
        #
        # # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg, img2_fg)
        image[o_x:o_x + rows, o_y:o_y + cols] = dst
        #

        count = count + 1
        # cv2.imwrite("result_landmark/%d.png" %(count),image)

        # cv2.imshow("lala", image)
        img_list = imagepath.split('.')
        new_path = img_list[0].split('/')[0] + '/target/' + img_list[0].split('/')[-1] + '-hair.jpg'
        print(new_path)
        cv2.imwrite(new_path, image)
        # cv2.waitKey(0)
    # cv2.imshow("lala", image)
    # cv2.waitKey(0)
