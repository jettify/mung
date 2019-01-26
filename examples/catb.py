import pandas as pd
import numpy as np

from itertools import combinations
from catboost import CatBoostClassifier


train_df = pd.read_csv('input/train.csv')
test_df  = pd.read_csv('input/test.csv')
print('read complete')

labels = train_df.target
test_id = test_df.ID

train_df.drop(['ID', 'target'], axis=1, inplace=True)
test_df.drop(['ID'], axis=1, inplace=True)

train_df.fillna(-9999, inplace=True)
test_df.fillna(-9999, inplace=True)

# Keep list of all categorical features in dataset to specify this for CatBoost
cat_features_ids = np.where(train_df.apply(pd.Series.nunique) < 30000)[0].tolist()

print('start training')
clf = CatBoostClassifier(
    learning_rate=0.1,
    iterations=1000,
    random_seed=0,
    #logging_level='Silent'
)
clf.fit(train_df, labels, cat_features=cat_features_ids)
print('stop training')

prediction = clf.predict_proba(test_df)[:, 1]

pd.DataFrame({'ID': test_id, 'PredictedProb': prediction}).to_csv('submission_base.csv', index=False)
