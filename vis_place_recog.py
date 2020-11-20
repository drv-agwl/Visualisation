import os as os
import json
import os as os
import numpy as np
from model import get_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd

place_coords = pd.read_csv('./robot_ground_truth.csv')

place_img_map = {"Place_0": "000000_rgb.png",
                 "Place_1": "000014_rgb.png",
                 "Place_2": "000018_rgb.png",
                 "Place_3": "000023_rgb.png",
                 "Place_4": "000028_rgb.png",
                 "Place_5": "000032_rgb.png",
                 "Place_6": "000051_rgb.png",
                 "Place_7": "000055_rgb.png",
                 "Place_8": "000056_rgb.png",
                 "Place_9": "000064_rgb.png",
                 "Place_10": "000035_rgb.png",
                 "Place_11": "000039_rgb.png",
                 "Place_12": "000043_rgb.png",
                 "Place_13": "000011_rgb.png"
                 }

dataset_dir = '../place_recognition/Complete_dataset'
test_dir = os.path.join(dataset_dir, 'test')


def get_place_coords():
    gt_test_labels = {}

    test_images = {}
    for place_id in os.listdir(test_dir):
        for img in os.listdir(os.path.join(test_dir, place_id)):
            gt_test_labels[img] = place_id
            test_images[img] = np.load(os.path.join(os.path.join(test_dir, place_id), img))

    test_size = len(gt_test_labels)

    ############################################################################################
    model = get_model(channels=10)

    opt = Adam(0.001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    model.load_weights('../place_recognition/Models/saved-model-99-0.94.hdf5')

    inputs = []
    for place_id, img in test_images.items():
        inputs.append(img)
    inputs = np.array(inputs)

    pred = model.predict(inputs, verbose=True)

    i = 0
    for place_id, img in test_images.items():
        test_images[place_id] = 'Place_' + str(np.argmax(pred[i], axis=-1))
        i += 1

    print(test_images)
    print(gt_test_labels)

    # plotting ground-truth places
    X = []
    Y = []
    for img_name, place_id in gt_test_labels.items():
        idx = int(place_img_map[place_id][:6])
        X.append(place_coords.loc[idx]["X"] / 100)
        Y.append(place_coords.loc[idx]["Z"] / 100)

    plt.scatter(X, Y)

    # plotting prediction places
    X = []
    Y = []
    for img_name, place_id in test_images.items():
        idx = int(place_img_map[place_id][:6])
        X.append(place_coords.loc[idx]["X"] / 100)
        Y.append(place_coords.loc[idx]["Z"] / 100)

    return X, Y
