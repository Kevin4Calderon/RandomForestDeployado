from flask import Flask, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier, plot_tree
from joblib import load
from sklearn.preprocessing import LabelEncoder
import os

# Configuración
app = Flask(__name__)
CORS(app)

# Cargar el modelo y el escalador entrenado
clf_tree_reduced = load('models/decision_tree_model_reduced.pkl')
scaler = load('models/scaler_reduced.pkl')

# Cargar los datos de un archivo CSV
data = pd.read_csv('data/archivo_reducido.csv')
X_train = data[['min_flowpktl', 'flow_fin']]
y_train = data['calss'] 

# Codificar etiquetas
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)

# Reducir y escalar los datos
X_train_reduced_scaled = scaler.transform(X_train)

@app.route('/')
def index():
    return render_template('index.html')  # Renderiza index.html

@app.route('/decision_boundary', methods=['GET'])
def decision_boundary():
    # Convertir los datos a numpy arrays
    X_train_reduced_np = X_train.values
    y_train_np = y_train_encoded

    # Verifica que los datos no contengan NaN o inf
    if np.any(np.isnan(X_train_reduced_np)) or np.any(np.isinf(X_train_reduced_np)):
        return jsonify({'error': 'Los datos contienen valores NaN o infinitos'}), 400

    # Crea la malla para graficar
    mins = X_train_reduced_np.min(axis=0) - 1
    maxs = X_train_reduced_np.max(axis=0) + 1
    x1, x2 = np.meshgrid(np.linspace(mins[0], maxs[0], 1000), np.linspace(mins[1], maxs[1], 1000))

    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf_tree_reduced.predict(X_new).reshape(x1.shape)

    # Graficar
    plt.figure(figsize=(12, 6))
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=custom_cmap)
    scatter = plt.scatter(X_train_reduced_np[:, 0], X_train_reduced_np[:, 1], 
                          c=y_train_np, edgecolors='k', cmap=custom_cmap, s=50, label='Entrenamiento')
    plt.xlim(mins[0], maxs[0])
    plt.ylim(mins[1], maxs[1])
    plt.xticks(np.arange(mins[0], maxs[0] + 1, 200))
    plt.yticks(np.arange(mins[1], maxs[1] + 1, 0.5))
    plt.xlabel('min_flowpktl', fontsize=14)
    plt.ylabel('flow_fin', fontsize=14, rotation=90)
    plt.title('Límite de Decisión', fontsize=16)
    plt.legend(*scatter.legend_elements(), title="Clases")
    plt.savefig('static/decision_boundary.png')
    plt.close()

    return jsonify({'message': 'Límite de decisión graficado!'})

@app.route('/decision_tree', methods=['GET'])
def decision_tree():
    # Usando plot_tree de scikit-learn para visualizar el árbol
    plt.figure(figsize=(12, 8))
    plot_tree(clf_tree_reduced, 
              filled=True, 
              feature_names=['min_flowpktl', 'flow_fin'], 
              class_names=label_encoder.classes_.tolist(),
              rounded=True)
    plt.savefig('static/decision_tree.png')  # Guardar el gráfico como PNG
    plt.close()
    
    return jsonify({'message': 'Árbol de decisión exportado!'})

@app.route('/dataset', methods=['GET'])
def dataset():
    # Visualización del dataset
    dataset_head = data.head().to_html(classes='table table-striped')
    dataset_info = data.info(buf=None)  # Captura el info en una variable
    dataset_info_html = f"<pre>{dataset_info}</pre>"  # Convierte a HTML
    value_counts = data["calss"].value_counts().to_frame().to_html(classes='table table-striped')

    return jsonify({
        'head': dataset_head,
        'info': dataset_info_html,
        'value_counts': value_counts
    })

@app.route('/scaled_values', methods=['GET'])
def scaled_values():
    X_train_scaled_df = pd.DataFrame(X_train_reduced_scaled, columns=X_train.columns)
    scaled_head = X_train_scaled_df.head(10).to_html(classes='table table-striped')
    return jsonify({'scaled_head': scaled_head})

@app.route('/f1_score', methods=['GET'])
def f1_score():
    from sklearn.metrics import f1_score
    y_train_pred = clf_tree_reduced.predict(X_train_reduced_scaled)
    f1 = f1_score(y_train_encoded, y_train_pred, average='weighted')
    return jsonify({'f1_score': f1})


# Ejecuta la aplicación Flask
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
