import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier


"""DATA PREPROCESSING"""
df = pd.read_csv("datasets/koi_cumulative.csv")

# 1 - Remove columns
# rowid
# kepid
# kepoi_name
# kepler_name
# koi_pdisposition
# koi_score
# koi_teq_err1
# koi_teq_err2
df.drop(columns=['kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition', 
                   'koi_score', 'koi_teq_err1', 'koi_teq_err2'], inplace=True)

# 2 - Limit values of target column.
# We left only 'confirmed' and 'false_positive'
candidate_rows = df.query("koi_disposition == 'CANDIDATE'").index
df = df.drop(candidate_rows, axis=0).reset_index(drop = True)

# 3 - Transform target column to binary
df['koi_disposition'] = df['koi_disposition'].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})

# 4 - Replace missing values
# Replace missing values in column 'koi_tce_delivname' with the mode
df['koi_tce_delivname'] = df['koi_tce_delivname'].fillna(df['koi_tce_delivname'].mode()[0])

# Replace missing values in all columns with the mean
for column in df.columns[df.isna().sum() > 0]:
    df[column] = df[column].fillna(df[column].mean())

# 5 - Convert categorical column 'koi_tce_delivname' into binary columns
delivname_dummies = pd.get_dummies(df['koi_tce_delivname'], prefix='delivname', dtype=int)
df = pd.concat([df, delivname_dummies], axis=1)
df = df.drop('koi_tce_delivname', axis=1)

# Save preprocessed data
df.to_csv("datasets/koi_processed.csv", index=False)



"""Gradient Boosted Tree Model"""

# 1 - Undersample to balance classes distribution
class_0 = df[df['koi_disposition'] == 0]
class_1 = df[df['koi_disposition'] == 1]

# Reduce the number of 'FALSE POSITIVE'
class_0_downsampled = class_0.sample(n=len(class_1), random_state=42)
df = pd.concat([class_0_downsampled, class_1])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# 2 - Train split
X = df.drop(columns=['koi_disposition'])
y = df['koi_disposition']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)

# 3 - Scale the data using a Standar Scaler
scaler = StandardScaler()
scaler.fit(X_train)

# Scaler.transform returns a numpy object, but it doesnt contain info about columns and index
# Thats why we transform it onto a dataframe
X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

# Dump the scaler
joblib.dump(scaler, 'models/KOI_model_scaler.pkl')

# 4 - Model Training

clf = GradientBoostingClassifier()

