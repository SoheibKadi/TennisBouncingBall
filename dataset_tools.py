import os
from typing import Dict

import cv2
import numpy as np
import pandas as pd

from motion_based_features import MotionBasedFeatures
from bounce_detector import BounceDetector

dataset_path = '../../TennisProject/data/dataset'
test_dataset_path = '../test_data'

def load_data(dataset_path=dataset_path, dataset_type='train'):
    if dataset_type == 'train':
        return load_data_train(dataset_path)
    else:
        return load_data_test(test_dataset_path)


def load_data_train(dataset_path):
    """
    Load the dataset from the given path
    """

    # Load images and labels
    # The dataset is split into games and each game is split into many clips.
    # each clip is a sequence of frames and the corresponding labels.
    #dataset = dict()
    bounce_detector = BounceDetector()
    motion_based_features = MotionBasedFeatures()

    num = 3
    colnames_x = ['x_diff_{}'.format(i) for i in range(1, num)] + \
                 ['x_diff_inv_{}'.format(i) for i in range(1, num)] + \
                 ['x_div_{}'.format(i) for i in range(1, num)]
    colnames_y = ['y_diff_{}'.format(i) for i in range(1, num)] + \
                 ['y_diff_inv_{}'.format(i) for i in range(1, num)] + \
                 ['y_div_{}'.format(i) for i in range(1, num)]
    colnames_area = ['area_diff_{}'.format(i) for i in range(1, num)] + \
                    ['area_diff_inv_{}'.format(i) for i in range(1, num)] + \
                    ['area_div_{}'.format(i) for i in range(1, num)]
    colnames_angle = ['angle_diff_{}'.format(i) for i in range(1, num)] + \
                     ['angle_diff_inv_{}'.format(i) for i in range(1, num)] + \
                     ['angle_div_{}'.format(i) for i in range(1, num)]
    colnames_size_1 = ['size_1_diff_{}'.format(i) for i in range(1, num)] + \
                      ['size_1_diff_inv_{}'.format(i) for i in range(1, num)] + \
                      ['size_1_div_{}'.format(i) for i in range(1, num)]
    colnames_size_2 = ['size_2_diff_{}'.format(i) for i in range(1, num)] + \
                      ['size_2_diff_inv_{}'.format(i) for i in range(1, num)] + \
                      ['size_2_div_{}'.format(i) for i in range(1, num)]
    colnames = colnames_x + colnames_y + colnames_area + colnames_angle + colnames_size_1 + colnames_size_2

    all_features = pd.DataFrame(columns=colnames)
    for game in os.listdir(dataset_path):
        game_path = os.path.join(dataset_path, game)
        if not os.path.isdir(game_path):
            continue

        #dataset[game]: Dict[str, pd.DataFrame] = dict()
        for clip in os.listdir(game_path):
            clip_path = os.path.join(game_path, clip)
            file_path = os.path.join(clip_path, 'label.csv')
            label = pd.read_csv(file_path)
            # Construct a new dataset by extracting features from the images
            frames = []
            previous_frame = None
            features = pd.DataFrame(columns=['frame_id', 'ball_x', 'ball_y', 'area', 'angle', 'size1', 'size2', 'bounce'])
            for i in range(len(label)):
                frame_path = os.path.join(clip_path, label.iloc[i]['file name'])
                visibility = label.iloc[i]['visibility']
                if visibility == 0:
                    if len(features) > 0:
                        f, _ = bounce_detector.prepare_features_new(features['ball_x'], features['ball_y'], features['area'], features['angle'], features['size1'], features['size2'], features['bounce'])
                        if len(f) > 0:
                            all_features = pd.concat([all_features, f])
                        features = pd.DataFrame(
                            columns=['frame_id', 'ball_x', 'ball_y', 'area', 'angle', 'size1', 'size2', 'bounce'])
                    previous_frame = cv2.imread(frame_path)
                    continue

                if previous_frame is None:
                    previous_frame = cv2.imread(frame_path)
                    continue

                frame = cv2.imread(frame_path)

                # Get the motion based feature
                feature = motion_based_features.get_ball_motion_features(previous_frame, frame, (label.iloc[i]['x-coordinate'], label.iloc[i]['y-coordinate']))

                features.loc[len(features)] = [i, label.iloc[i]['x-coordinate'], label.iloc[i]['y-coordinate'], feature[0], feature[1], feature[2], feature[3], label.iloc[i]['status'] == 2]

                previous_frame = frame

            if len(features) > 0:
                f, _ = bounce_detector.prepare_features_new(features['ball_x'], features['ball_y'], features['area'],
                                                            features['angle'], features['size1'], features['size2'], features['bounce'])
                if len(f) > 0:
                    all_features = pd.concat([all_features, f])

            #dataset[game][clip] = features

    return all_features


def load_data_test(dataset_path):
    """
    Load the dataset from the given path
    """

    # Load images and labels
    # The dataset is split into games and each game is split into many clips.
    # each clip is a sequence of frames and the corresponding labels.
    bounce_detector = BounceDetector()
    motion_based_features = MotionBasedFeatures()

    num = 3
    colnames_x = ['x_diff_{}'.format(i) for i in range(1, num)] + \
                 ['x_diff_inv_{}'.format(i) for i in range(1, num)] + \
                 ['x_div_{}'.format(i) for i in range(1, num)]
    colnames_y = ['y_diff_{}'.format(i) for i in range(1, num)] + \
                 ['y_diff_inv_{}'.format(i) for i in range(1, num)] + \
                 ['y_div_{}'.format(i) for i in range(1, num)]
    colnames_area = ['area_diff_{}'.format(i) for i in range(1, num)] + \
                    ['area_diff_inv_{}'.format(i) for i in range(1, num)] + \
                    ['area_div_{}'.format(i) for i in range(1, num)]
    colnames_angle = ['angle_diff_{}'.format(i) for i in range(1, num)] + \
                     ['angle_diff_inv_{}'.format(i) for i in range(1, num)] + \
                     ['angle_div_{}'.format(i) for i in range(1, num)]
    colnames_size_1 = ['size_1_diff_{}'.format(i) for i in range(1, num)] + \
                      ['size_1_diff_inv_{}'.format(i) for i in range(1, num)] + \
                      ['size_1_div_{}'.format(i) for i in range(1, num)]
    colnames_size_2 = ['size_2_diff_{}'.format(i) for i in range(1, num)] + \
                      ['size_2_diff_inv_{}'.format(i) for i in range(1, num)] + \
                      ['size_2_div_{}'.format(i) for i in range(1, num)]
    colnames = colnames_x + colnames_y + colnames_area + colnames_angle + colnames_size_1 + colnames_size_2

    all_features = pd.DataFrame(columns=colnames)
    for game in os.listdir(dataset_path):
        if 'mp4' not in game or 'Predicted' in game:
            continue

        cap = cv2.VideoCapture(os.path.join(dataset_path, game))
        if not cap.isOpened():
            print("Error opening video file")
            continue

        bouncing_file_path = os.path.join(dataset_path, game.replace('.mp4', '_out.txt'))
        bouncing_file = open(bouncing_file_path, 'r')
        bounces = bouncing_file.readlines()
        bouncing_file.close()

        ball_coordinates_file = open(os.path.join(dataset_path, game.replace('.mp4', '_coordinates.txt')), 'r')
        ball_coordinates = ball_coordinates_file.readlines()
        ball_coordinates_file.close()

        frame_id = 0
        previous_frame = None
        features = pd.DataFrame(columns=['frame_id', 'ball_x', 'ball_y', 'area', 'angle', 'size1', 'size2', 'bounce'])
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Construct a new dataset by extracting features from the images
            x_coordinate, y_coordinate = map(int, ball_coordinates[frame_id].split(' '))
            if x_coordinate == 0 and y_coordinate == 0:
                if len(features) > 0:
                    f, _ = bounce_detector.prepare_features_new(features['ball_x'], features['ball_y'],
                                                                features['area'], features['angle'], features['size1'],
                                                                features['size2'], features['bounce'])
                    if len(f) > 0:
                        all_features = pd.concat([all_features, f])
                    features = pd.DataFrame(
                        columns=['frame_id', 'ball_x', 'ball_y', 'area', 'angle', 'size1', 'size2', 'bounce'])
                previous_frame = frame
                frame_id += 1
                continue

            if previous_frame is None:
                previous_frame = frame
                frame_id += 1
                continue

            # Get the motion based feature
            feature = motion_based_features.get_ball_motion_features(previous_frame, frame, (x_coordinate, y_coordinate))

            features.loc[len(features)] = [frame_id, x_coordinate, y_coordinate, feature[0], feature[1], feature[2], feature[3], int(bounces[frame_id])]

            previous_frame = frame
            frame_id += 1

        if len(features) > 0:
            f, _ = bounce_detector.prepare_features_new(features['ball_x'], features['ball_y'], features['area'],
                                                        features['angle'], features['size1'], features['size2'],
                                                        features['bounce'])
            if len(f) > 0:
                all_features = pd.concat([all_features, f])

    return all_features


def test_load_data():
    dataset = load_data()
    assert len(dataset) > 0
    assert len(dataset['game1']) > 0
    assert len(dataset['game1']['Clip1']) > 0
    assert len(dataset['game1']['Clip1']['frame_id']) > 0
    assert len(dataset['game1']['Clip1']['ball_x']) > 0
    assert len(dataset['game1']['Clip1']['ball_y']) > 0
    assert len(dataset['game1']['Clip1']['area']) > 0
    assert len(dataset['game1']['Clip1']['angle']) > 0
    assert len(dataset['game1']['Clip1']['size1']) > 0
    assert len(dataset['game1']['Clip1']['size2']) > 0
    assert len(dataset['game1']['Clip1']['bounce']) > 0
    print('Test passed')


#if __name__ == '__main__':
#test_load_data()