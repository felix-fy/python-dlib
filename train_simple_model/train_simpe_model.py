# -*- coding: utf-8 -*-
import os
import sys
import glob
import dlib
import cv2

# set option
options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = True
options.C = 5
options.num_threads = 16
options.be_verbose = True

# set photo path
current_path = os.getcwd()
train_folder = current_path + '/photo/train/'
test_folder = current_path + '/photo/test/'
train_xml_path = train_folder + 'train.xml'
test_xml_path = test_folder + 'test.xml'

# train model
print("start training:")
dlib.train_simple_object_detector(train_xml_path, 'detector.svm', options)

# test model
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(train_xml_path, "detector.svm")))
print("Testing accuracy: {}".format(
    dlib.test_simple_object_detector(test_xml_path, "detector.svm")))
