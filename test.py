from tkinter import *
from PIL import Image, ImageTk
import cv2
import combine


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


def take_snapshot(path):
    print("有人给你点赞啦！")

    # video_loop(img_path)
    img_path = '/Users/dani/Codes/workspaces_AI/hair/data/hairs/hair02.jpg'
    print(img_path)


class CanvasDemo:
    def __init__(self):
        self. window = Tk()
        self.window.title("CanvasDemo")
        self.camera = cv2.VideoCapture(0)  # 摄像头
        self.panel = Label(self.window, height=300, width=300)  # initialize image panel
        self.panel.pack(padx=100, pady=100)
        self. window.config(cursor="arrow")

        self.canvas = Canvas(self.window, width=500, height=500, bg="White")
        self.canvas.pack()

        frame = Frame(self.window)
        frame.pack()

        img_path = '/Users/dani/Codes/workspaces_AI/hair/data/hairs/hair01.jpg'
        im = Image.open(img_path)
        w, h = im.size
        pil_image_resized = resize(w, h, 80, 80, im)
        tk_image = ImageTk.PhotoImage(pil_image_resized)

        btRectangle = Button(frame, text="长方形", command=lambda: take_snapshot(img_path), image=tk_image)
        btOval = Button(frame, text="椭 圆", command=lambda: take_snapshot(img_path), image=tk_image)
        btArc = Button(frame, text="圆 弧", command=lambda: take_snapshot(img_path), image=tk_image)
        btPolygon = Button(frame, text="多边形", command=lambda: take_snapshot(img_path), image=tk_image)
        btLine = Button(frame, text=" 线 ", command=lambda: take_snapshot(img_path), image=tk_image)
        btString = Button(frame, text="文 字", command=lambda: take_snapshot(img_path), image=tk_image)
        btClear = Button(frame, text="清 空", command=lambda: take_snapshot(img_path), image=tk_image)

        btRectangle.grid(row=1, column=1)
        btOval.grid(row=1, column=2)
        btArc.grid(row=1, column=3)
        btPolygon.grid(row=1, column=4)
        btLine.grid(row=1, column=5)
        btString.grid(row=1, column=6)
        btClear.grid(row=1, column=7)

        self.window.mainloop()

    #
    # def displayRect(self):
    #     self.canvas.create_rectangle(10, 10, 190, 90, tags = "rect")
    # def displayOval(self):
    #     self.canvas.create_oval(10, 10, 190, 90, tags = "oval", fill = "red")
    # def displayArc(self):
    #     self.canvas.create_arc(10, 10, 190, 90, start = 0, extent = 90, width = 8, fill = "red", tags = "arc")
    # def displayPolygon(self):
    #     self.canvas.create_polygon(10, 10, 190, 90, 30, 50, tags = "polygon")
    # def displayLine(self):
    #     self.canvas.create_line(10, 10, 190, 90, fill = 'red', tags = "line")
    #     self.canvas.create_line(10, 90, 190, 10, width = 9, arrow = "last", activefill = "blue", tags = "line")
    # def displayString(self):
    #     self.canvas.create_text(60, 40, text = "Hi,i am a string", font = "Tine 10 bold underline", tags = "string")
    # def clearCanvas(self):
    #     self.canvas.delete("rect", "oval", "arc", "polygon", "line", "string")

    def video_loop(self, img_path=None):
        # print('-----',img_path)
        success, img = self.camera.read()  # 从摄像头读取照片

        if success:
            cv2.waitKey(100)

            imgtk = combine.combine(img, img_path)
            print(imgtk)
            print(imgtk.shape)

            cv2image = cv2.cvtColor(imgtk, cv2.COLOR_BGR2RGBA)  # 转换颜色从BGR到RGBA
            current_image = Image.fromarray(cv2image)  # 将图像转换成Image对象
            imgtk = ImageTk.PhotoImage(image=current_image)
            # imgtk = ImageTk.PhotoImage(image=Image.fromqpixmap(imgtk))

            # print(type(current_image))
            # print(current_image.shape)
            # print(current_image[0, 0, :])

            self.panel.imgtk = imgtk
            self. panel.config(image=imgtk)
            self.window.after(1, lambda: self.video_loop(img_path))
            # print(img_path)


CanvasDemo()
