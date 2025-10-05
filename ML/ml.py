import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64
from io import BytesIO
from sklearn.metrics import confusion_matrix


def train_gbt(
        df,
        target_column,
        n_estimators,
        learning_rate,
        max_depth,
        min_samples_split,
        train_size=0.7,
        scaler_type = 'standard' # standard or minmax
):
    # Train split
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=True, random_state=1)

    scaler = None

    # Scale X
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    X_train = pd.DataFrame(scaler_type.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test = pd.DataFrame(scaler_type.transform(X_test), index=X_test.index, columns=X_test.columns)

    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, min_samples_split=min_samples_split)

    model.fit(X_train, y_train)

    # Evaluation
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    matrix = confusion_matrix_graph(y_test, prediction)
    
    #Graphs
    matrix_graph = confusion_matrix_graph(y_test,prediction)
    bar = bar_chart(accuracy,recall,precision,f1)
    feature_imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    feature_importance = feature_importance_graph(feature_imp)

    graphics = {'confussion_matrix': matrix_graph,
                'feature_importance': feature_importance,
                'metrics_bar': bar}
    
    return model, scaler, graphics


def predict(X, model, scaler):
    X_scaled = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    positive_probs = probabilities[:, 1]

    results = {'predictions': predictions, 'probabilities': positive_probs}
    return results

    


def confusion_matrix_graph(y_test, y_pred):
    cf_matrix = confusion_matrix(y_test, y_pred)
    group_names = ['True Neg','False Pos','False Neg','True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    #ax.set_title('Confusion Matrix with labels\n\n');
    ax.set_xlabel('Values ​​predicted by the model')
    ax.set_ylabel('Real values')

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