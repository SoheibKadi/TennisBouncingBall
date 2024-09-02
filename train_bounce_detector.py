import catboost as cb
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from dataset_tools import load_data
from bounce_detector import BounceDetector

# Check if the dataset is already created
try:
    X = np.load('X.npy')
    y = np.load('y.npy')
except:
        dataset = load_data()

        # Train a catboost model for bounce detection
        y = dataset['bounce'].to_numpy()
        X = dataset.drop(columns=['bounce']).to_numpy()

        # Save to file
        np.save('X.npy', X)
        np.save('y.npy', y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

train_dataset = cb.Pool(X_train, y_train)
test_dataset = cb.Pool(X_test, y_test)

model = cb.CatBoostRegressor(learning_rate=0.1, iterations=2000, loss_function='RMSE')
model.fit(train_dataset, eval_set=test_dataset, verbose=True)

pred = model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, pred)))
r2 = r2_score(y_test, pred)
print('Testing performance')
print('RMSE: {:.2f}'.format(rmse))
print('R2: {:.2f}'.format(r2))

# Save the model
model.save_model('models/bounce_detector.cbm')

df_test_dataset = load_data(dataset_type='test')
y_df_test = df_test_dataset['bounce'].to_numpy()
X_df_test = df_test_dataset.drop(columns=['bounce']).to_numpy()

test_dataset = cb.Pool(X_df_test, y_df_test)
pred = model.predict(test_dataset)
rmse = (np.sqrt(mean_squared_error(y_df_test, pred)))
r2 = r2_score(y_df_test, pred)
print('Testing performance On DreamFight dataset')
print('RMSE: {:.2f}'.format(rmse))
print('R2: {:.2f}'.format(r2))


# Caluculating Precision and Recall for the model using pred and y_df_test
def calculate_precision_recall(y_df_test, pred):
    y_df_test = y_df_test.astype(int)
    pred = pred.astype(int)

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    print(f'Dataset size {len(y_df_test)}')

    for i in range(len(y_df_test)):
        if y_df_test[i] == 1 and pred[i] == 1:
            tp += 1
        elif y_df_test[i] == 0 and pred[i] == 1:
            fp += 1
        elif y_df_test[i] == 1 and pred[i] == 0:
            fn += 1
        elif y_df_test[i] == 0 and pred[i] == 0:
            tn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall


precision, recall = calculate_precision_recall(y_df_test, pred)
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
# Calculate the F1 score
f1 = 2 * (precision * recall) / (precision + recall)
print('F1 score: {:.2f}'.format(f1))




