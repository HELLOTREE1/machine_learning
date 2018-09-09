# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import cv2
img=cv2.imread("/home/szu/PycharmProjects/machineLearning/Logistic/迭代次数于回归系数.jpg",cv2.IMREAD_COLOR)
plt.show(img)
# print(cv2.__version__)
# capture = cv2.CaptureFromFile('/media/szu/HELLOTREE/课时10Numpy基础结构.mp4')
# nbFrames = int(cv2.GetCaptureProperty(capture, cv2.CV_CAP_PROP_FRAME_COUNT))
#
# # CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream
# # CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream
#
# fps = cv2.GetCaptureProperty(capture, cv2.CV_CAP_PROP_FPS)
#
# wait = int(1 / fps * 1000 / 1)
#
# duration = (nbFrames * fps) / 1000
#
# print('Num. Frames = ', nbFrames)
# print('Frame Rate = ', fps, 'fps')
# print('Duration = ', duration, 'sec')