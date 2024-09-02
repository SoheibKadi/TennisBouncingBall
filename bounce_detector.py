import catboost as ctb
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import distance

class BounceDetector:
    def __init__(self, path_model=None, is_new_model=False):
        self.model = ctb.CatBoostRegressor()
        self.threshold = 0.45
        self.is_new_model = is_new_model
        if path_model:
            self.load_model(path_model)
        
    def load_model(self, path_model):
        self.model.load_model(path_model)
    
    def prepare_features(self, x_ball, y_ball):
        labels = pd.DataFrame({'frame': range(len(x_ball)), 'x-coordinate': x_ball, 'y-coordinate': y_ball})
        
        num = 3
        eps = 1e-15
        for i in range(1, num):
            labels['x_lag_{}'.format(i)] = labels['x-coordinate'].shift(i)
            labels['x_lag_inv_{}'.format(i)] = labels['x-coordinate'].shift(-i)
            labels['y_lag_{}'.format(i)] = labels['y-coordinate'].shift(i)
            labels['y_lag_inv_{}'.format(i)] = labels['y-coordinate'].shift(-i) 
            labels['x_diff_{}'.format(i)] = abs(labels['x_lag_{}'.format(i)] - labels['x-coordinate'])
            labels['y_diff_{}'.format(i)] = labels['y_lag_{}'.format(i)] - labels['y-coordinate']
            labels['x_diff_inv_{}'.format(i)] = abs(labels['x_lag_inv_{}'.format(i)] - labels['x-coordinate'])
            labels['y_diff_inv_{}'.format(i)] = labels['y_lag_inv_{}'.format(i)] - labels['y-coordinate']
            labels['x_div_{}'.format(i)] = abs(labels['x_diff_{}'.format(i)]/(labels['x_diff_inv_{}'.format(i)] + eps))
            labels['y_div_{}'.format(i)] = labels['y_diff_{}'.format(i)]/(labels['y_diff_inv_{}'.format(i)] + eps)

        for i in range(1, num):
            labels = labels[labels['x_lag_{}'.format(i)].notna()]
            labels = labels[labels['x_lag_inv_{}'.format(i)].notna()]
        labels = labels[labels['x-coordinate'].notna()] 
        
        colnames_x = ['x_diff_{}'.format(i) for i in range(1, num)] + \
                     ['x_diff_inv_{}'.format(i) for i in range(1, num)] + \
                     ['x_div_{}'.format(i) for i in range(1, num)]
        colnames_y = ['y_diff_{}'.format(i) for i in range(1, num)] + \
                     ['y_diff_inv_{}'.format(i) for i in range(1, num)] + \
                     ['y_div_{}'.format(i) for i in range(1, num)]
        colnames = colnames_x + colnames_y

        features = labels[colnames]
        return features, list(labels['frame'])

    def prepare_features_new(self, x_ball, y_ball, area, angle, size_1, size_2, bounce=None):

        labels = pd.DataFrame({'frame': range(len(x_ball)), 'x-coordinate': x_ball, 'y-coordinate': y_ball, 'area': area, 'angle': angle, 'size_1': size_1, 'size_2': size_2})
        # Add bounce column
        if bounce is not None:
            labels['bounce'] = bounce

        num = 3
        eps = 1e-15
        for i in range(1, num):
            labels['x_lag_{}'.format(i)] = labels['x-coordinate'].shift(i)
            labels['x_lag_inv_{}'.format(i)] = labels['x-coordinate'].shift(-i)
            labels['y_lag_{}'.format(i)] = labels['y-coordinate'].shift(i)
            labels['y_lag_inv_{}'.format(i)] = labels['y-coordinate'].shift(-i)
            labels['x_diff_{}'.format(i)] = abs(labels['x_lag_{}'.format(i)] - labels['x-coordinate'])
            labels['y_diff_{}'.format(i)] = labels['y_lag_{}'.format(i)] - labels['y-coordinate']
            labels['x_diff_inv_{}'.format(i)] = abs(labels['x_lag_inv_{}'.format(i)] - labels['x-coordinate'])
            labels['y_diff_inv_{}'.format(i)] = labels['y_lag_inv_{}'.format(i)] - labels['y-coordinate']
            labels['x_div_{}'.format(i)] = abs(
                labels['x_diff_{}'.format(i)] / (labels['x_diff_inv_{}'.format(i)] + eps))
            labels['y_div_{}'.format(i)] = labels['y_diff_{}'.format(i)] / (labels['y_diff_inv_{}'.format(i)] + eps)
            labels['area_lag_{}'.format(i)] = labels['area'].shift(i)
            labels['area_lag_inv_{}'.format(i)] = labels['area'].shift(-i)
            labels['area_diff_{}'.format(i)] = abs(labels['area_lag_{}'.format(i)] - labels['area'])
            labels['area_diff_inv_{}'.format(i)] = abs(labels['area_lag_inv_{}'.format(i)] - labels['area'])
            labels['area_div_{}'.format(i)] = abs(
                labels['area_diff_{}'.format(i)] / (labels['area_diff_inv_{}'.format(i)] + eps))
            labels['angle_lag_{}'.format(i)] = labels['angle'].shift(i)
            labels['angle_lag_inv_{}'.format(i)] = labels['angle'].shift(-i)
            labels['angle_diff_{}'.format(i)] = abs(labels['angle_lag_{}'.format(i)] - labels['angle'])
            labels['angle_diff_inv_{}'.format(i)] = abs(labels['angle_lag_inv_{}'.format(i)] - labels['angle'])
            labels['angle_div_{}'.format(i)] = abs(
                labels['angle_diff_{}'.format(i)] / (labels['angle_diff_inv_{}'.format(i)] + eps))
            labels['size_1_lag_{}'.format(i)] = labels['size_1'].shift(i)
            labels['size_1_lag_inv_{}'.format(i)] = labels['size_1'].shift(-i)
            labels['size_1_diff_{}'.format(i)] = abs(labels['size_1_lag_{}'.format(i)] - labels['size_1'])
            labels['size_1_diff_inv_{}'.format(i)] = abs(labels['size_1_lag_inv_{}'.format(i)] - labels['size_1'])
            labels['size_1_div_{}'.format(i)] = abs(
                labels['size_1_diff_{}'.format(i)] / (labels['size_1_diff_inv_{}'.format(i)] + eps))
            labels['size_2_lag_{}'.format(i)] = labels['size_2'].shift(i)
            labels['size_2_lag_inv_{}'.format(i)] = labels['size_2'].shift(-i)
            labels['size_2_diff_{}'.format(i)] = abs(labels['size_2_lag_{}'.format(i)] - labels['size_2'])
            labels['size_2_diff_inv_{}'.format(i)] = abs(labels['size_2_lag_inv_{}'.format(i)] - labels['size_2'])
            labels['size_2_div_{}'.format(i)] = abs(
                labels['size_2_diff_{}'.format(i)] / (labels['size_2_diff_inv_{}'.format(i)] + eps))

        for i in range(1, num):
            labels = labels[labels['x_lag_{}'.format(i)].notna()]
            labels = labels[labels['x_lag_inv_{}'.format(i)].notna()]
            labels = labels[labels['area_lag_{}'.format(i)].notna()]
            labels = labels[labels['area_lag_inv_{}'.format(i)].notna()]
            labels = labels[labels['angle_lag_{}'.format(i)].notna()]
            labels = labels[labels['angle_lag_inv_{}'.format(i)].notna()]
            labels = labels[labels['size_1_lag_{}'.format(i)].notna()]
            labels = labels[labels['size_1_lag_inv_{}'.format(i)].notna()]
            labels = labels[labels['size_2_lag_{}'.format(i)].notna()]
            labels = labels[labels['size_2_lag_inv_{}'.format(i)].notna()]
            if 'bounce' in labels.columns:
                labels = labels[labels['bounce'].notna()]

        labels = labels[labels['x-coordinate'].notna()]

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
        if 'bounce' in labels.columns:
            colnames.append('bounce')

        features = labels[colnames]
        return features, list(labels['frame'])
    
    def predict(self, x_ball, y_ball, smooth=True):
        if smooth:
            x_ball, y_ball = self.smooth_predictions(x_ball, y_ball)
        features, num_frames = self.prepare_features(x_ball, y_ball)
        preds = self.model.predict(features)
        ind_bounce = np.where(preds > self.threshold)[0]
        if len(ind_bounce) > 0:
            ind_bounce = self.postprocess(ind_bounce, preds)
        frames_bounce = [num_frames[x] for x in ind_bounce]
        return set(frames_bounce)

    def predict_new(self, x_ball, y_ball, area, angle, size_1, size_2):
        # if smooth:
        #     x_ball, y_ball = self.smooth_predictions(x_ball, y_ball)
        features, num_frames = self.prepare_features_new(x_ball, y_ball, area, angle, size_1, size_2)
        preds = self.model.predict(features)
        ind_bounce = np.where(preds > self.threshold)[0]
        if len(ind_bounce) > 0:
            ind_bounce = self.postprocess(ind_bounce, preds)
        frames_bounce = [num_frames[x] for x in ind_bounce]
        return set(frames_bounce)
    
    def smooth_predictions(self, x_ball, y_ball):
        is_none = [int(x is None) for x in x_ball]
        interp = 5
        counter = 0
        for num in range(interp, len(x_ball)-1):
            if not x_ball[num] and sum(is_none[num-interp:num]) == 0 and counter < 3:
                x_ext, y_ext = self.extrapolate(x_ball[num-interp:num], y_ball[num-interp:num])
                x_ball[num] = x_ext
                y_ball[num] = y_ext
                is_none[num] = 0
                if x_ball[num+1]:
                    dist = distance.euclidean((x_ext, y_ext), (x_ball[num+1], y_ball[num+1]))
                    if dist > 80:
                        x_ball[num+1], y_ball[num+1], is_none[num+1] = None, None, 1
                counter += 1
            else:
                counter = 0  
        return x_ball, y_ball

    def extrapolate(self, x_coords, y_coords):
        xs = list(range(len(x_coords)))
        func_x = CubicSpline(xs, x_coords, bc_type='natural')
        x_ext = func_x(len(x_coords))
        func_y = CubicSpline(xs, y_coords, bc_type='natural')
        y_ext = func_y(len(x_coords))
        return float(x_ext), float(y_ext)    

    def postprocess(self, ind_bounce, preds):
        ind_bounce_filtered = [ind_bounce[0]]
        for i in range(1, len(ind_bounce)):
            if (ind_bounce[i] - ind_bounce[i-1]) != 1:
                cur_ind = ind_bounce[i]
                ind_bounce_filtered.append(cur_ind)
            elif preds[ind_bounce[i]] > preds[ind_bounce[i-1]]:
                ind_bounce_filtered[-1] = ind_bounce[i]
        return ind_bounce_filtered


