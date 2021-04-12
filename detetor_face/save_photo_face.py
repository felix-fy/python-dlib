# -*- coding utf-8 -*-
import dlib         
import numpy as np  
import cv2          
import os
import sys

# 读取图像的路径
path_read = "/process/photo_dir/"
dirs = os.listdir( path_read )

# 用来存储生成的单张人脸的路径
path_save = "/process/face_dir/"

# Dlib 预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

def main ():
    for path_file in dirs:
        #path_file = "u=649350073,2266937680&fm=27&gp=0.jpg"
        img = cv2.imread(path_read+path_file)
    
        # Dlib 检测
        faces = detector(img, 1)
    
        #print(path_file+"face number:", len(faces), '\n')
    
        for k, d in enumerate(faces):
    
            # 计算矩形大小
            # (x,y), (宽度width, 高度height)
            pos_start = tuple([d.left(), d.top()])
            pos_end = tuple([d.right(), d.bottom()])

            # 计算矩形框大小
            height = d.bottom()-d.top()
            width = d.right()-d.left()

            # 根据人脸大小生成空的图像
            img_blank = np.zeros((height, width, 3), np.uint8)

            for i in range(height):
                for j in range(width):
                    img_blank[i][j] = img[d.top()+i][d.left()+j]

            # 存在本地
            #print("Save to:", path_save+path_file+".face"+str(k+1)+".jpg")
            cv2.imwrite(path_save+path_file+".face"+str(k+1)+".jpg", img_blank)

if __name__ == '__main__':
    main()
