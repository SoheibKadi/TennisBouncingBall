import os
import cv2
import numpy as np
import pandas as pd

from motion_based_features import MotionBasedFeatures
from bounce_detector import BounceDetector

dataset_path = '../../TennisProject/data/dataset'


def load_data(dataset_path=dataset_path):
    """
    Load the dataset from the given path
    """

    # Load images and labels
    # The dataset is split into games and each game is split into many clips.
    # each clip is a sequence of frames and the corresponding labels.
    dataset = dict()
    bounce_detector = BounceDetector()
    motion_based_features = MotionBasedFeatures()
    for game in os.listdir(dataset_path):
        game_path = os.path.join(dataset_path, game)
        if not os.path.isdir(game_path):
            continue

        dataset[game] = dict()
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
                    continue

                if previous_frame is None:
                    previous_frame = cv2.imread(frame_path)
                    continue

                frame = cv2.imread(frame_path)

                # Get the motion based feature
                feature = motion_based_features.get_ball_motion_features(previous_frame, frame, (label.iloc[i]['x-coordinate'], label.iloc[i]['y-coordinate']))

                features.loc[len(features)] = [i, label.iloc[i]['x-coordinate'], label.iloc[i]['y-coordinate'], feature[0], feature[1], feature[2], feature[3], label.iloc[i]['status'] == 2]

                previous_frame = frame

            dataset[game][clip] = features

    return dataset


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
test_load_data()