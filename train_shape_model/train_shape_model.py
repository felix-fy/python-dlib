# -*- coding: utf-8 -*-
import os
import sys
import dlib

# set option
train_folder = 'photo/train'
test_folder = 'photo/test'
options = dlib.shape_predictor_training_options()
options.oversampling_amount = 300
options.nu = 0.15
options.tree_depth = 3
options.be_verbose = True

# train model
training_xml_path = os.path.join(train_folder, "training_with_face_landmarks.xml")
dlib.train_shape_predictor(training_xml_path, "predictor.dat", options)
print("\nTraining accu\fracy: {}".format(
    dlib.test_shape_predictor(training_xml_path, "predictor.dat")))

# test model
testing_xml_path = os.path.join(test_folder, "testing_with_face_landmarks.xml")
print("Testing accu\fracy: {}".format(
    dlib.test_shape_predictor(testing_xml_path, "predictor.dat")))
