import cv2
from opencv_dnn import detectFaceOpenCVDnn

way_flag = 1        #0使用默认xml识别，1使用DNN识别

if __name__ == '__main__':
    # 加载图像
    cap = cv2.VideoCapture(0)  # 可以将这里换成你的视频所在的地址
    # 加载人脸检测器模型
    face_detector = cv2.CascadeClassifier('./data/haarcascades/haarcascade_frontalface_default.xml')
    while True:
        retval, image = cap.read()  # retval 为返回的布尔值，有图片时返回True，否则为False

        #image = cv2.resize(image, (960,540))  # 减少图像尺寸减少计算量，这里可以根据你的视频中图像的大小进行调整

        if not retval:  # 当读取到最后一帧图片时，退出，停止读取
            break

        if(way_flag == 0):
            gray = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)  # 转化为灰度图进行检测，减少计算量
            faces = face_detector.detectMultiScale(gray)  # 获得人脸检测结果
            for x,y,w,h in faces:  # 进行for循环，将检测到的所有人脸区域绘制标注方框
                cv2.rectangle(image, pt1=(x,y), pt2=(x+w,y+h), color=[0,0,255], thickness=2)
        if (way_flag == 1):
            faces = detectFaceOpenCVDnn(image)
            for x,y,w,h in faces:  # 进行for循环，将检测到的所有人脸区域绘制标注方框
                cv2.rectangle(image, pt1=(x,y), pt2=(w,h), color=[0,0,255], thickness=2)
        cv2.imshow('image', image)
        key = cv2.waitKey(1)
        if key == ord('q'):  # 当输入q时，停止读取
            break
    #print(image.shape)
    cap.release()  # 释放内存
