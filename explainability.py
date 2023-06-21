import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# Load the data from the CSV files
drivers_data = 'C:/Users/William/Dropbox/MPDI/2023/Formula 1/f1db_csv/drivers.csv'
qualifying_data = 'C:/Users/William/Dropbox/MPDI/2023/Formula 1/f1db_csv/qualifying.csv'
lap_times_data = 'C:/Users/William/Dropbox/MPDI/2023/Formula 1/f1db_csv/lap_times.csv'

# Load the trained model
model = load_model('modelo_entrenado.h5')

# Step 4: Application of explainability techniques (for example, attention)
# Load the data from the CSV files
drivers = pd.read_csv(drivers_data)
qualifying = pd.read_csv(qualifying_data)
lap_times = pd.read_csv(lap_times_data)

# Perform the union of the data using the appropriate columns
# (make any necessary adjustments based on your data structure)
data = pd.merge(drivers, qualifying, on='driverId', how='inner')
data = pd.merge(data, lap_times, on=['raceId', 'driverId'], how='inner')

# Split the data into training and test sets
X = data[['driverId', 'position_x', 'lap']]
y = data['position_x']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data preprocessing
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check the dimensions of X_train_scaled
print("Dimensiones de X_train_scaled:", X_train_scaled.shape)

# Reshape the input data
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))


# Apply care techniques
layer_outputs = [layer.output for layer in model.layers]
attention_model = Model(inputs=model.input, outputs=layer_outputs)


attention_outputs = attention_model.predict(X_train_reshaped)

# Get attention weights
attention_weights = attention_outputs[-2]
#print("Dimensiones de attention_weights antes de np.squeeze:", attention_weights.shape)

# Apply np.squeeze to remove axis 2 if necessary
if attention_weights.ndim > 2:
    attention_weights = np.squeeze(attention_weights, axis=2)

#print("Dimensiones de attention_weights después de np.squeeze:", attention_weights.shape)


# Visualize attention weights as heat maps
plt.imshow(attention_weights.T, cmap='hot', interpolation='nearest')
plt.xlabel('Time Step')
plt.ylabel('Feature')
plt.colorbar()
plt.show()


# Calculate feature importance using Permutation Importance
def permutation_importance(model, X, y, metric, n_iterations=100):
    baseline_score = metric(y, model.predict(X))
    feature_importances = []
    for feature in range(X.shape[1]):
        X_permuted = X.copy()
        X_permuted[:, feature] = np.random.permutation(X_permuted[:, feature])
        permuted_score = metric(y, model.predict(X_permuted))
        feature_importance = baseline_score - permuted_score
        feature_importances.append(feature_importance)
    return np.array(feature_importances)

# Calculate feature importance using Permutation Importance
#importance_scores = permutation_importance(model, X_test_reshaped, y_test, metric=mean_squared_error)
# Calculate feature importance using Permutation Importance
importance_scores = permutation_importance(model, X_test_reshaped, y_test, metric=mean_squared_error)

# Check the length of importance_scores
print("Longitud de importance_scores:", len(importance_scores))

# Display feature importances
feature_names = X.columns
sorted_indices = np.argsort(importance_scores)
plt.barh(range(len(feature_names)), importance_scores[sorted_indices], tick_label=feature_names[sorted_indices])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

print("Dimensiones de importance_scores:", importance_scores.shape)

# Remodelar X_test_reshaped para que tenga la misma dimensión que X_train
X_test_reshaped = X_test_reshaped.reshape((X_test_reshaped.shape[0], X_test_reshaped.shape[1], 1))

# Check the length of predictions and importance_scores
predictions = model.predict(X_test_reshaped)

print("Longitud de predictions:", len(predictions))
print("Longitud de importance_scores:", len(importance_scores))

# Compare predictions and explanations
for i in range(min(len(predictions), len(importance_scores))):
    print(f"Prediction: {predictions[i]}, Importance: {importance_scores[i]}")



