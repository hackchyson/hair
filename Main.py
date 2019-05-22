from tkinter import *
import cv2
from PIL import Image, ImageTk
import combine
import numpy as np

img_path = '/Users/dani/Codes/workspaces_AI/hair/data/hairs/hair01.jpg'


def take_snapshot(path):
    print("有人给你点赞啦！")

    # video_loop(img_path)
    img_path = '/Users/dani/Codes/workspaces_AI/hair/data/hairs/hair02.jpg'
    print(img_path)


def video_loop(img_path=None):
    # print('-----',img_path)
    success, img = camera.read()  # 从摄像头读取照片
    print('-'*30,img.shape)
    # print(img.shape)
    # print(img[0, 0, :])
    if success:
        cv2.waitKey(100)
        # print(type(img))
        # print(img.shape)
        # print(img[0, 0, :])
        # cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)#转换颜色从BGR到RGBA
        # current_image = Image.fromarray(cv2image)#将图像转换成Image对象

        imgtk = combine.combine(img, img_path)
        print(imgtk)
        print(imgtk.shape)
        # b_channel, g_channel, r_channel = cv2.split(imgtk)
        #
        # alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 125
        # cv2image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

        # print(type(imgtk))
        # print(imgtk.shape)
        # print(imgtk[0, 0, :])
        cv2image = cv2.cvtColor(imgtk, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA
        current_image = Image.fromarray(cv2image)  # 将图像转换成Image对象
        imgtk = ImageTk.PhotoImage(image=current_image)
        # imgtk = ImageTk.PhotoImage(image=Image.fromqpixmap(imgtk))

        # print(type(current_image))
        # print(current_image.shape)
        # print(current_image[0, 0, :])

        panel.imgtk = imgtk
        panel.config(image=imgtk)
        root.after(1, lambda: video_loop(img_path))
        # print(img_path)


def resize(w, h, w_box, h_box, pil_image):
    '''
    resize a pil_image object so it will fit into
    a box of size w_box times h_box, but retain aspect ratio
    对一个pil_image对象进行缩放，让它在一个矩形框内，还能保持比例
    '''
    f1 = 1.0 * w_box / w  # 1.0 forces float division in Python2
    f2 = 1.0 * h_box / h
    factor = min([f1, f2])
    # print(f1, f2, factor) # test
    # use best down-sizing filter
    width = int(w * factor)
    height = int(h * factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)


camera = cv2.VideoCapture(0)  # 摄像头

root = Tk()
root.geometry("1000x1000")
root.title("opencv + tkinter")
# root.protocol('WM_DELETE_WINDOW', detector)

panel = Label(root, height=300, width=300)  # initialize image panel
panel.pack(padx=100, pady=100)
root.config(cursor="arrow")

im = Image.open(img_path)
# img = ImageTk.PhotoImage(im)


w, h = im.size
pil_image_resized = resize(w, h, 80, 80, im)
tk_image = ImageTk.PhotoImage(pil_image_resized)

btn = Button(root, text="点赞!", command=lambda: take_snapshot(img_path), image=tk_image, width=10, height=10)
btn.pack(fill="both", expand=True, padx=10, pady=10)

bt_start = Button(root, text="点赞!", command=lambda: take_snapshot(img_path), image=tk_image, width=10, height=10)
bt_start.pack(fill="both", expand=True, padx=10, pady=10)

# bt_start = Button(root, text='获取视频', height=2, width=15, command=video_loop)
# bt_start.place(x=1, y=1)
# bt_start = Button(root, text="获取视频", command=take_snapshot)
# bt_start.pack(fill="both", expand=True, padx=10, pady=10)

video_loop(img_path)

root.mainloop()
# 当一切都完成后，关闭摄像头并释放所占资源
camera.release()
cv2.destroyAllWindows()
