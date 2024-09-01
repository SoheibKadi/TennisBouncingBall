import cv2
import numpy as np


class MotionBasedFeatures:
    def __init__(self, threshold=15, blur=False, blur_kernel_size=(5, 5)):
        """
        Creates the motion based features extractor.
        :param threshold: Threshold value for the motion detection.
        :param blur: If True, apply Gaussian blur to the frames before motion detection.
        :param blur_kernel_size: The kernel size for the Gaussian blur.
        """
        self.threshold = threshold
        self.blur = blur
        self.blur_kernel_size = blur_kernel_size

    def get_ball_motion_features(self, previous_frame, current_frame, ball_position):
        frame_diff = MotionBasedFeatures.__motion_detection(previous_frame, current_frame, self.threshold, self.blur, self.blur_kernel_size)
        ball_blob = MotionBasedFeatures.__get_ball_blob(frame_diff, ball_position)
        return MotionBasedFeatures.__get_blob_features(ball_blob)

    @staticmethod
    def __motion_detection(previous_frame, current_frame, threshold, blur, blur_kernel_size):
        # Convert the frames to grayscale
        previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Guassian blur the frames
        if blur:
            previous_frame_gray = cv2.GaussianBlur(previous_frame_gray, blur_kernel_size, 0)
            current_frame_gray = cv2.GaussianBlur(current_frame_gray, blur_kernel_size, 0)

        # Get the difference between the two frames
        frame_diff = cv2.absdiff(previous_frame_gray, current_frame_gray)

        # Apply a threshold to the difference
        _, frame_diff = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)

        return frame_diff

    @staticmethod
    def __get_closest_blob(contours, ball_position):
        """
        Get the closest blob to the ball position
        :param contours:
        :param ball_position: Tuple of the ball position (x, y)
        :return:
        """
        # get the keypoint with the closest center to the ball position
        min_distance = float('inf')
        closest_keypoint = (ball_position, (0, 0), 0)
        for contour in contours:
            # get the distance between the keypoint and the ball position
            # if the distance is less than the ball radius, return the keypoint
            keypoint = cv2.minAreaRect(contour)
            if keypoint[1][0] == 0 or keypoint[1][1] == 0:
                continue
            keypoint_center = keypoint[0]

            distance = np.linalg.norm(np.array(keypoint_center) - np.array(ball_position))
            if distance < min_distance:
                min_distance = distance
                closest_keypoint = keypoint

        return closest_keypoint

    @staticmethod
    def __get_ball_blob(frame_diff, ball_position):
        # Find the contours in the ball patch
        contours, _ = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get the closest blob to the ball position
        ball_blob = MotionBasedFeatures.__get_closest_blob(contours, ball_position)

        return ball_blob

    @staticmethod
    def __get_blob_features(blob):
        # Get the center and area of the blob
        center, size, angle = blob
        area = size[0] * size[1]

        # normalize the angle to be between 0 and 180 degrees
        angle = angle if angle < 180 else angle - 180

        if size[0] < size[1]:
            size_big = size[1]
            size_small = size[0]
        else:
            size_big = size[0]
            size_small = size[1]

        return center[0], center[1], area, angle, size_big, size_small