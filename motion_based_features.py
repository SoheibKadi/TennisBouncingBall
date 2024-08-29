import cv2

import numpy as np


def motion_detection(previous_frame, current_frame):
    # Convert the frames to grayscale
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Guassian blur the frames
    #previous_frame_gray = cv2.GaussianBlur(previous_frame_gray, (5, 5), 0)
    #current_frame_gray = cv2.GaussianBlur(current_frame_gray, (5, 5), 0)

    # Get the difference between the two frames
    frame_diff = cv2.absdiff(previous_frame_gray, current_frame_gray)

    # Apply a threshold to the difference
    _, frame_diff = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)

    return frame_diff


def get_ball_motion_patch(frame_diff, ball_position):
    # Get the bounding box of the ball
    x, y, w, h = ball_position

    # Extend the bounding box by 2 times the ball radius
    x -= w
    y -= h
    w += 2 * w
    h += 2 * h

    # Extract the ball patch from the frame difference
    ball_patch = frame_diff[y:y+h, x:x+w]

    return ball_patch, (x, y, w, h)


def get_closest_blob(contours, ball_position):
    # get the keypoint with the closest center to the ball position
    for contour in contours:
        # get the distance between the keypoint and the ball position
        # if the distance is less than the ball radius, return the keypoint
        keypoint = cv2.minAreaRect(contour)
        keypoint_center = keypoint.center
        ball_center = (ball_position[0] + ball_position[2] / 2, ball_position[1] + ball_position[3] / 2)
        ball_radius = min(ball_position[2], ball_position[3]) / 2
        distance = np.linalg.norm(np.array(keypoint_center) - np.array(ball_center))
        if distance < ball_radius:
            return keypoint

    return None


def get_ball_blob(frame_diff, ball_position):
    ball_patch, patch_offset = get_ball_motion_patch(frame_diff, ball_position)
    # Find tcontours in the ball patch
    contours, _ = cv2.findContours(ball_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Correct ball position using patch offset
    ball_position = (ball_position[0] - patch_offset[0], ball_position[1] - patch_offset[1], ball_position[2], ball_position[3])

    # Get the closest blob to the ball position
    ball_blob = get_closest_blob(contours, ball_position)

    return ball_blob


def get_blob_features(blob):
    # Get the center and area of the blob
    center, size, angle = blob
    area = size[0] * size[1]

    # normalize the angle to be between 0 and 180 degrees
    angle = angle if angle < 180 else angle - 180

    return center[0], center[1], area, angle, size[0], size[1]

