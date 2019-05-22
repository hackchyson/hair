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

# hair location config
json_config = "config/hair.json"
with open(json_config, "r") as json_file:
    data = json.load(json_file)

    x_shift_ratio = data['x_shift_ratio']
    y_shift_ratio = data['y_shift_ratio']
    x_widen = data["x_widen"]
    y_widen = data['y_widen']

# model config
test_mode = "ONet"
thresh = [0.9, 0.6, 0.7]

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

#######################


gt_imdb = []

path = "data/source"

for item in os.listdir(path):
    if ('jpg' not in item):
        continue
    gt_imdb.append(os.path.join(path, item))

# gt_imdb = ['data/hairs/obama.mp4']


hair = cv2.imread('data/hairs/hair01.jpg')
rows, cols, channels = hair.shape


#

def get_video(path, hair):
    # videoCapture = cv2.VideoCapture(path)
    videoCapture = cv2.VideoCapture(0)
    # cv2.cam
    # 读帧
    success, frame = videoCapture.read()
    # print(type(frame))
    # print(frame.shape)
    f_h, f_w, f_c = frame.shape
    cont = 0
    while success:
        if cont % 2 == 0:
            all_boxes, landmarks = mtcnn_detector.detect(frame)
        # cont += 1
        # # time.sleep(0.1)
        for bbox in all_boxes:
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 7)
            # cv2.putText(frame, str(np.round(bbox[4], 2)), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_TRIPLEX, 1,
            #             color=(255, 0, 255))
            dst = (int((bbox[2] - bbox[0]) * x_widen), int((bbox[3] - bbox[1]) * y_widen))
            hair = cv2.imread('data/hairs/hair01.jpg')
            hair = cv2.resize(hair, dst, 0, 0, cv2.INTER_LANCZOS4)
            # hair = cv2.pyrUp(hair,dst)
            rows, cols, channels = hair.shape

            o_x = int(bbox[1] - (bbox[3] - bbox[1]) * x_shift_ratio)
            o_y = int(bbox[0] - (bbox[2] - bbox[0]) * y_shift_ratio)

            cv2.putText(frame, str(int(o_x)) + ',' + str(int(o_y)), (0, 50),
                        cv2.FONT_HERSHEY_TRIPLEX, 1,
                        color=(255, 0, 255))

            if o_x < 0 or o_y < 0 or o_x + rows > f_h or o_y + cols > f_w:
                # print("o_x: {}; o_y: {}".format(o_x, o_y))
                break

            # cv2.rectangle(frame, (o_x, o_y), (o_x + cols, o_y + rows), (0, 0, 255), 7)

            # Now create a mask of logo and create its inverse mask also
            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 这个254很重要
            mask_inv = cv2.bitwise_not(mask, dst=None, mask=None)
            #
            # cv2.imshow('mask', mask_inv)
            # Now black-out the area of logo in ROI
            # print(rows, cols, channels)
            #
            roi = frame[o_x:o_x + rows, o_y:o_y + cols]

            print("o_x: {}; o_y: {}".format(o_x, o_y))
            print(frame.shape)
            print(hair.shape)
            print('box: ', bbox)
            print('roi', roi.shape)
            #
            img2_fg = cv2.bitwise_and(hair, hair, mask=mask_inv)  # 这里才是mask_inv
            img1_bg = cv2.bitwise_and(roi, roi, mask=mask)

            #
            # # Take only region of logo from logo frame.
            #
            # # Put logo in ROI and modify the main frame
            dst = cv2.add(img1_bg, img2_fg)
            frame[o_x:o_x + rows, o_y:o_y + cols] = dst

        # for bbox in [0, 0, 100, 100]:
        # cv2.rectangle(frame, (0, 0), (100, 100), (100, 100, 100), 7)
        cv2.imshow("Oto Video", frame)  # 显示
        cv2.waitKey(40)  # 延迟
        success, frame = videoCapture.read()  # 获取下一帧


def read_img(gt_imdb):
    test_data = TestLoader(gt_imdb)
    all_boxes, landmarks = mtcnn_detector.detect_face(test_data)
    count = 0
    for imagepath in gt_imdb:
        image = cv2.imread(imagepath)

        hair = cv2.imread('data/hairs/1111.png')
        # rows, cols, channels = hair.shape

        for bbox, landmark in zip(all_boxes[count], landmarks[count]):
            # for bbox in all_boxes:
            cv2.putText(image, str(int(bbox[0])) + ',' + str(int(bbox[1])), (0, 100),
                        cv2.FONT_HERSHEY_TRIPLEX, 1,
                        color=(255, 0, 255))

            print(bbox)

            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)

            ############################ 3
            hair = cv2.resize(hair, (int((bbox[2] - bbox[0]) * x_widen), int((bbox[3] - bbox[1]) * y_widen)))
            rows, cols, channels = hair.shape

            o_x = int(bbox[1] - (bbox[3] - bbox[1]) * x_shift_ratio)
            o_y = int(bbox[0] - (bbox[2] - bbox[0]) * y_shift_ratio)

            cv2.putText(image, str(int(o_x)) + ',' + str(int(o_y)), (0, 50),
                        cv2.FONT_HERSHEY_TRIPLEX, 1,
                        color=(255, 0, 255))

            if o_x < 0 or o_y < 0 or o_x + rows > 250 or o_y + cols > 250:
                print("o_x: {}; o_y: {}".format(o_x, o_y))
                break

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

            # cv2.imwrite("result_landmark/%d.png" %(count),image)

            cv2.imshow("lala", image)
            img_list = imagepath.split('.')
            new_path = img_list[0].split('/')[0] + '/target/' + img_list[0].split('/')[-1] + '-hair.jpg'
            # print(new_path)
            # cv2.imwrite(new_path, image)
            print('image shape: ', image.shape)
            cv2.waitKey(0)
        count = count + 1


get_video('data/hairs/obama.mp4', hair)
# read_img(gt_imdb)
