# 导入opencv模块
import cv2
import numpy as np
import os
import shutil
import time


#自定义配置参数
VedioAddress = 'http://192.168.1.165:4747/mjpegfeed?600x480'
haarcascade_address = r"haarcascades/haarcascade_frontalface_alt.xml"
picture_size = (600,480)


#采集自己的人脸数据
def generator(data):
    '''
    打开摄像头，读取帧，检测该帧图像中的人脸，并进行剪切、缩放
    生成图片满足以下格式：
    1.灰度图，后缀为 .png
    2.图像大小相同
    params:
        data:指定生成的人脸数据的保存路径
    '''
    #计数
    count = 1

    # 加载人脸模型，字符串是文件路径
    face = cv2.CascadeClassifier(haarcascade_address)

    # 打开摄像头
    capture = cv2.VideoCapture(VedioAddress)

    # 获取摄像头实时画面
    cv2.namedWindow("face")
    while True:
        # 读取摄像头的帧画面。ret的值为True或False,代表有没有读到图片，frame是当前截取一帧的图片
        ret, frame = capture.read()

        # 调整图片灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # 检查人脸
        faces = face.detectMultiScale(gray, 1.1, 5, 0)

        # 标记人脸
        for (x, y, w, h) in faces:
            # 里面有四个参数 1图片 2坐标原点 3识别大小 4颜色RGB 5线宽
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 显示图片，渲染画面
            cv2.imshow('face', frame)
            #sz = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            img = cv2.resize(frame, picture_size, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(r"face/rcz/rcz%s.jpg"%(str(count)), img)
            count += 1

            # 暂停窗口
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
            if count >= 100:
                break
         # 暂停窗口
        if count >= 100:
            break
    # 释放资源
    capture.release()

    # 5关闭窗口
    cv2.destroyAllWindows()


# 载入图像   读取ORL人脸数据库，准备训练数据
def LoadImages(data):
    '''
    加载图片数据用于训练
    params:
        data:训练数据所在的目录，要求图片尺寸一样
    ret:
        images:[m,height,width]  m为样本数，height为高，width为宽
        names：名字的集合
        labels：标签
    '''
    images = []
    names = []
    labels = []

    label = 0

    # 遍历所有文件夹
    for subdir in os.listdir(data):
        subpath = os.path.join(data,  subdir)
        print('path',subpath)
        # 判断文件夹是否存在
        if os.path.isdir(subpath):
            # 在每一个文件夹中存放着一个人的许多照片
            names.append(subdir)
            # 遍历文件夹中的图片文件
            for filename in os.listdir(subpath):
                imgpath = os.path.join(subpath, filename)
                img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
                img = cv2.resize(img, picture_size, interpolation=cv2.INTER_LINEAR)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #cv2.imshow('1',img)
                #cv2.waitKey(0)
                images.append(gray_img)
                labels.append(label)
            label += 1
    images = np.asarray(images)
    # names=np.asarray(names)
    labels = np.asarray(labels)
    return images, labels, names


# 检验训练结果
def FaceRec(data):
    # 加载训练的数据
    X, y, names = LoadImages(data)
    #print('x',X)
    # 人脸识别的模型
    model = cv2.face.EigenFaceRecognizer_create()
    # fisherfaces算法的模型
    #model = cv2.face.FisherFaceRecognizer_create()
    # LBPH算法的模型
    #model = cv2.face.LBPHFaceRecognizer_create()
    model.train(X, y)
    model.save(r"haarcascades/train")
    model.read(r"haarcascades/train")
    print('train over')
    # 打开摄像头
    camera = cv2.VideoCapture(VedioAddress)
    cv2.namedWindow('face')

    # 创建级联分类器
    face_casecade = cv2.CascadeClassifier(haarcascade_address)

    while (True):
        # 读取一帧图像
        # ret:图像是否读取成功
        # frame：该帧图像
        ret, frame = camera.read()
        # 判断图像是否读取成功
        # print('ret',ret)
        if ret:
            # 转换为灰度图
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 利用级联分类器鉴别人脸
            faces = face_casecade.detectMultiScale(gray_img, 1.1, 5)

            # 遍历每一帧图像，画出矩形
            for (x, y, w, h) in faces:
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 蓝色
                roi_gray = gray_img[y:y + h, x:x + w]

                try:
                    # 将图像转换为宽92 高112的图像
                    # resize（原图像，目标大小，（插值方法）interpolation=，）
                    roi_gray = cv2.resize(roi_gray, picture_size, interpolation=cv2.INTER_LINEAR)
                    params = model.predict(roi_gray)
                    print('Label:%s,confidence:%.2f' % (names[params[0]], params[1]))
                    '''
                    putText:给照片添加文字
                    putText(输入图像，'所需添加的文字'，左上角的坐标，字体，字体大小，颜色，字体粗细)
                    '''
                    cv2.putText(frame, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                except:
                    continue

            cv2.imshow('face', frame)

            # 按下q键退出
            if cv2.waitKey(100) & 0xff == ord('q'):
                break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    start = time.perf_counter()
    data = './face'
    generator(data)
    FaceRec(data)
    elapsed = (time.perf_counter() - start)
    print("Time used:", elapsed)