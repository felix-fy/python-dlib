# coding:utf-8:q

import dlib
import cv2
import glob
import numpy as np
import os
import sys
import pymysql

class face_recognition:
    def __init__(self,predictor_path,face_rec_model_path):
        self.predictor_path = predictor_path
        self.face_rec_model_path = face_rec_model_path
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(self.predictor_path)
        self.face_rec_model = dlib.face_recognition_model_v1(self.face_rec_model_path)
    def face_detection(self,url_img_1,url_img_2):
        img_path_list = [url_img_1,url_img_2]
        dist = []
        for img_path in img_path_list:
            img = cv2.imread(img_path)
            # 转换rgb顺序的颜色。
            b, g, r = cv2.split(img)
            img2 = cv2.merge([r, g, b])
            # 检测人脸
            faces = self.detector(img, 1)
            if len(faces):
                for index, face in enumerate(faces):
                    # # 提取68个特征点
                    shape = self.shape_predictor(img2, face)
                    # 计算人脸的128维的向量
                    face_descriptor = self.face_rec_model.compute_face_descriptor(img2, shape)
                    dist.append(list(face_descriptor))
            else:
                pass
        return dist
    # 欧式距离
    def dist_o(self,dist_1,dist_2):
        dis = np.sqrt(sum((np.array(dist_1)-np.array(dist_2))**2))
        return dis
    def score(self,url_img_1,url_img_2):
        url_img_1 = glob.glob(url_img_1)[0]
        url_img_2 = glob.glob(url_img_2)[0]
        data = self.face_detection(url_img_1,url_img_2)
        goal = self.dist_o(data[0],data[1])
        #print ('dis2=%s' % (goal))
        # 判断结果，如果goal小于0.6的话是同一个人，否则不是。我所用的是欧式距离判别
        if goal > 0.0:
            return 1-goal
        else:
            return 1-goal
def main ():
    # 调用 模型下载地址：http://dlib.net/files/
    predictor_path = "./files/shape_predictor_68_face_landmarks.dat"
    face_rec_model_path = "./files/dlib_face_recognition_resnet_model_v1.dat"
    face_ = face_recognition(predictor_path,face_rec_model_path)
    path_read = "./result_photo/"
    dirs = os.listdir( path_read )
    numones = 1

    db = pymysql.connect("10.10.180.43","root","123456","dlib" )
    cursor = db.cursor()

    for img_a in dirs:
        img_1 = path_read+img_a
        print(numones,'>== now analysis is '+img_a+' ==<')
        for img_b in dirs:
            img_2 = path_read+img_b
            #print(img_2)
            goal = face_.score(img_1,img_2)
            print(goal)
            sql = "INSERT INTO face_data(num_id,a_photo_name,b_photo_name,contrast_data) \
                    VALUES ('%s','%s','%s','%s')" % (numones,img_a,img_b,goal)
            try:
                cursor.execute(sql)
                db.commit()
            except:
                db.rollback()
        numones = numones + 1
    
    db.close()
    
if __name__ == '__main__':
    main()
