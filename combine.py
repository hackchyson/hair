from MtcnnDetector import MtcnnDetector
from detector import Detector
from fcn_detector import FcnDetector
from mtcnn_model import P_Net, R_Net, O_Net
import cv2
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
# print(model_path)
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

frame_buf = None


def combine(frame, path):
    f_h, f_w, f_c = frame.shape
    all_boxes, _ = mtcnn_detector.detect(frame)
    global frame_buf
    if frame is not None:
        frame_buf = frame
    else:
        # print('222222222222222')
        frame = frame_buf

    if len(all_boxes) >= 1:
        for bbox in all_boxes:
            dst = (int((bbox[2] - bbox[0]) * x_widen), int((bbox[3] - bbox[1]) * y_widen))
            hair = cv2.imread(path)
            hair = cv2.resize(hair, dst, 0, 0, cv2.INTER_LANCZOS4)
            rows, cols, channels = hair.shape
            hair_width = rows
            hair_height = cols

            o_x = int(bbox[1] - (bbox[3] - bbox[1]) * x_shift_ratio)
            o_y = int(bbox[0] - (bbox[2] - bbox[0]) * y_shift_ratio)

            cv2.putText(frame, str(int(o_x)) + ',' + str(int(o_y)), (0, 50),
                        cv2.FONT_HERSHEY_TRIPLEX, 1,
                        color=(255, 0, 255))

            if o_x < 0 or o_y < 0 or o_x + rows > f_h  or o_y + cols > f_w :
                # print("o_x: {}; o_y: {}".format(o_x, o_y))
                break
            # print('dani'*3)
            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 这个254很重要
            mask_inv = cv2.bitwise_not(mask, dst=None, mask=None)

            # Now black-out the area of logo in ROI
            # print(rows, cols, channels)
            roi = frame[o_x:o_x + rows, o_y:o_y + cols]

            # print("o_x: {}; o_y: {}".format(o_x, o_y))11111
            # print(frame.shape)
            # print(hair.shape)
            # print('box: ', bbox)
            # print('roi', roi.shape)

            img2_fg = cv2.bitwise_and(hair, hair, mask=mask_inv)  # 这里才是mask_inv
            img1_bg = cv2.bitwise_and(roi, roi, mask=mask)

            #
            # # Put logo in ROI and modify the main frame
            dst = cv2.add(img1_bg, img2_fg)
            frame[o_x:o_x + rows, o_y:o_y + cols] = dst

        if frame is None:
            print('1111111')
            return frame_buf
        return frame
    else:
        return frame
