import pandas as pd
import numpy as np

from itertools import combinations
from catboost import CatBoostClassifier


train_df = pd.read_csv('input/train.csv')
test_df  = pd.read_csv('input/test.csv')
print('read complete')
labels = train_df.target
test_id = test_df.ID


selected_features = [
    'v10', 'v12', 'v14', 'v21', 'v22', 'v24', 'v30', 'v31', 'v34', 'v38', 'v40', 'v47', 'v50',
    'v52', 'v56', 'v62', 'v66', 'v72', 'v75', 'v79', 'v91', 'v112', 'v113', 'v114', 'v129'
]

# drop some of the features that were not selected
train_df = train_df[selected_features]
test_df = test_df[selected_features]
import ipdb
ipdb.set_trace()

train_df.fillna(-9999, inplace=True)
test_df.fillna(-9999, inplace=True)

# update the list of categorical features
cat_features_ids = np.where(train_df.apply(pd.Series.nunique) < 30000)[0].tolist()

char_features = list(train_df.columns[train_df.dtypes == np.object])
char_features_without_v22 = list(train_df.columns[(train_df.dtypes == np.object) & (train_df.columns != 'v22')])

cmbs = list(combinations(char_features, 2)) + list(map(lambda x: ("v22",) + x, combinations(char_features_without_v22, 2)))

def concat_columns(df, columns):
    value = df[columns[0]].astype(str) + ' '
    for col in columns[1:]:
        value += df[col].astype(str) + ' '
    return value

# add new features based on combinations/interactions
for cols in cmbs:
    train_df["".join(cols)] = concat_columns(train_df, cols)
    test_df["".join(cols)] = concat_columns(test_df, cols)

# add new engineered features to the list of categorical features in dataframe
cat_features_ids += range(len(selected_features), train_df.shape[1])
clf = CatBoostClassifier(
    learning_rate=0.1, iterations=1000, random_seed=0,
    # logging_level='Silent'
)

print('start training')
clf.fit(train_df, labels, cat_features=cat_features_ids)

prediction = clf.predict_proba(test_df)[:,1]

pd.DataFrame(
        {'ID':test_id, 'PredictedProb':prediction}
).to_csv(
        'submission_improved.csv', index=False
    )
