import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.datasets import load_iris  # Import Iris dataset loader
import pickle
import os

# Define absolute paths based on the docker-compose.yaml volumes
BASE_DIR = "/opt/airflow/working_data"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Define file paths
TRAIN_FILE_PATH = os.path.join(DATA_DIR, "file.csv")  # <-- ADD THIS LINE
TEST_FILE_PATH = os.path.join(DATA_DIR, "test.csv")
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "model.sav")
SCALER_FILE_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.

    Returns:
        bytes: Serialized data.
    """
    # Create data directory if it doesn't exist (for robustness)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # This now reads the file.csv you just created
    df = pd.read_csv(TRAIN_FILE_PATH)
    serialized_data = pickle.dumps(df)
    
    return serialized_data
    

def data_preprocessing(data):
    """
    Deserializes data, performs data preprocessing, saves the scaler,
    and returns serialized clustered data.

    Args:
        data (bytes): Serialized data to be deserialized and processed.

    Returns:
        bytes: Serialized clustered data.
    """
    df = pickle.loads(data)
    df = df.dropna()
    # The DataFrame already contains only the features we need
    
    min_max_scaler = MinMaxScaler()
    # Fit and transform the entire DataFrame
    clustering_data_minmax = min_max_scaler.fit_transform(df)
    
    # --- Save the scaler object for later use in prediction ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(SCALER_FILE_PATH, 'wb') as f:
        pickle.dump(min_max_scaler, f)
    # --- End Save ---

    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return clustering_serialized_data


def build_save_model(data, filename):
    """
    Builds a KMeans clustering model, saves it to a file, and returns SSE values.

    Args:
        data (bytes): Serialized data for clustering.
        filename (str): Name of the file to save the clustering model.

    Returns:
        list: List of SSE (Sum of Squared Errors) values for different numbers of clusters.
    """
    df = pickle.loads(data)
    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    
    sse = []
    # Note: The loop saves the last model (k=49), not the one from the elbow.
    # This matches your original logic.
    for k in range(1, 50):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
    
    # Create the model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    output_path = os.path.join(MODEL_DIR, filename)

    # Save the trained model (k=49) to a file
    with open(output_path, 'wb') as f:
        pickle.dump(kmeans, f)
        
    return sse

def load_model_elbow(filename, sse):
    """
    Loads a saved KMeans model, scales test data, and predicts.
    Also determines the number of clusters using the elbow method and prints it.

    Args:
        filename (str): Name of the file containing the saved clustering model.
        sse (list): List of SSE values for different numbers of clusters.

    Returns:
        int: The predicted cluster for the first test sample.
    """
    
    # Load the saved model
    model_path = os.path.join(MODEL_DIR, filename)
    loaded_model = pickle.load(open(model_path, 'rb'))

    # Load test data
    df = pd.read_csv(TEST_FILE_PATH)
    
    # --- Load the saved scaler and transform the test data ---
    with open(SCALER_FILE_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    # Ensure test data columns match the training data
    df_scaled = scaler.transform(df)
    # --- End Fix ---
    
    kl = KneeLocator(
        range(1, 50), sse, curve="convex", direction="decreasing"
    )

    # Optimal clusters (Note: this is just printed)
    print(f"Optimal no. of clusters: {kl.elbow}")

    # Make predictions on the SCALED test data
    predictions = loaded_model.predict(df_scaled)
    
    # Return the prediction for the first sample, as in the original file
    return predictions[0]