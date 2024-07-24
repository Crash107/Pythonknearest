import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Assuming you have a DataFrame named 'df' with your data
# For simplicity, let's consider only 'danceability_%' and 'streams' columns

# Load your dataset into a pandas DataFrame
# df = pd.read_csv('your_dataset.csv')

df = pd.read_csv('spotify-2023.csv', encoding='latin1')

# Convert 'streams' to numeric, handle missing values
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
df = df.dropna(subset=['streams'])

# Select input features and target variable


# Select relevant columns
selected_columns = ['danceability_%', 'streams']
df_selected = df[selected_columns]

# Drop rows with missing values
df_selected = df_selected.dropna()

# Split the dataset into features (X) and target variable (y)
X = df_selected[['danceability_%']]
y = df_selected['streams']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the k-NN model
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = knn.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test['danceability_%'], y=y_test, label='Actual Streams')
sns.scatterplot(x=X_test['danceability_%'], y=y_pred, label='Predicted Streams')
plt.xlabel('Danceability %')
plt.ylabel('Number of Streams')
plt.title('Scatter Plot of Danceability vs. Streams')
plt.legend()
plt.show()
