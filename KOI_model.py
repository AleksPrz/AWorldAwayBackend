import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64
from io import BytesIO
from sklearn.metrics import confusion_matrix

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
joblib.dump(scaler, 'models/KOI_scaler.pkl')

# 4 - Model Training

clf = GradientBoostingClassifier()

clf.fit(X_train, y_train)

# Evaluate model
prediction = clf.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
recall = recall_score(y_test, prediction)
precision = precision_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
matrix = confusion_matrix(y_test, prediction)

key_params = ['learning_rate', 'max_depth', 'min_samples_split', 'n_estimators', 'subsample']
params = clf.get_params()
filtered_params = {p: params[p] for p in key_params}

# Guardar en JSON
with open("models/KOI_best_params.json", "w") as f:
    json.dump(filtered_params, f, indent=4)

print(f'Accuracy: {accuracy}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print('Confusion Matrix:')
print(matrix)

joblib.dump(clf, 'models/KOI.pkl')

# Graphics

matrix = confusion_matrix(y_test, prediction)
bar = bar_chart(accuracy,recall,precision,f1)
feature_imp = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_importance = feature_importance_graph(feature_imp)

def confusion_matrix(y_test, y_pred):
    cf_matrix = confusion_matrix(y_test, y_pred)
    group_names = ['True Neg','False Pos','False Neg','True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    #ax.set_title('Confusion Matrix with labels\n\n');
    ax.set_xlabel('Valores preditos pelo modelo')
    ax.set_ylabel('Valores reais ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    
    buffer = BytesIO()
    plt.savefig(buffer, format='jpeg')
    buffer.seek(0)
    img_bytes = buffer.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return img_base64

def feature_importance_graph(feature_imp):
    plt.figure(figsize=(10,6))
    sns.barplot(x=feature_imp, y=feature_imp.index, palette="viridis")
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.tight_layout()

    # Convertir a base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_bytes = buffer.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return img_base64

def bar_chart(accuracy, recall,precision,f1):
    # Tus métricas
    metrics = {
        "Accuracy": accuracy,
        "Recall": recall,
        "Precision": precision,
        "F1 Score": f1
    }

    names = list(metrics.keys())
    values = list(metrics.values())

    # Crear la figura
    plt.figure(figsize=(8,5))
    sns.barplot(x=names, y=values, palette="Blues_r")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Evaluation Metrics of the Model")

    # Mostrar valores encima de las barras
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

    plt.tight_layout()

    # Convertir la gráfica a bytes y luego a base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()  # Cerrar la figura para liberar memoria
    buffer.seek(0)
    img_bytes = buffer.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    return img_base64
