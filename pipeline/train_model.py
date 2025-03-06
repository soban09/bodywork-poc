import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from datetime import datetime
import pickle, os, logging, json

# Set up logging
logger = logging.getLogger()

def load():
    logger.info("Loading the dataset...")
    data = pd.read_csv('./pipeline/dataset/diabetes-dev.csv')
    logger.info("Dataset loaded")
    return data

def handle_missing_data(data):
    data.fillna(data.mean(), inplace=True)
    return data

def min_max_scaling(data):
    scaler = MinMaxScaler()
    data[[
        'PlasmaGlucose', 
        'DiastolicBloodPressure', 
        'TricepsThickness', 
        'SerumInsulin', 
        'BMI', 
        'DiabetesPedigree', 
        'Age']
        ] = scaler.fit_transform(data[
            ['PlasmaGlucose', 
                'DiastolicBloodPressure', 
                'TricepsThickness', 
                'SerumInsulin', 
                'BMI', 
                'DiabetesPedigree', 
                'Age']
            ])
    
    return data

def handle_feature_scaling(data):
    scaler = StandardScaler()
    data[
        ['PlasmaGlucose', 
            'DiastolicBloodPressure', 
            'TricepsThickness', 
            'SerumInsulin', 
            'BMI', 
            'DiabetesPedigree', 
            'Age']
        ] = scaler.fit_transform(data[[
            'PlasmaGlucose', 
            'DiastolicBloodPressure', 
            'TricepsThickness', 
            'SerumInsulin', 
            'BMI', 
            'DiabetesPedigree', 
            'Age']
            ])
    return data

def preprocess(data):
    logger.info("Handling missing data...")
    data = handle_missing_data(data)
    
    logger.info("Handling Feature Scaling...")
    data = min_max_scaling(data)
    
    logger.info("Handling Min-Max Scaling...")
    data = handle_feature_scaling(data)
    
    return data

def split(data):
    logger.info("Splitting the dataset...")
    
    X = data.drop(columns=['Diabetic', 'PatientID'])
    y = data['Diabetic']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    logger.info("Dataset splitted")

    return X_train, X_test, y_train, y_test

def train(X_train, y_train):
    logger.info("Training the model...")
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    logger.info("Model trained")

    return model

def evaluate(model, X_test, y_test):
    # Evaluate the model
    logger.info("Evaluating the model...")

    y_pred = model.predict(X_test)
    accuracy =  accuracy_score(y_test, y_pred),
    precision = precision_score(y_test, y_pred),
    recall = recall_score(y_test, y_pred),
    _f1_score = f1_score(y_test, y_pred),
    _confusion_matrix = confusion_matrix(y_test, y_pred).tolist()

    logger.info(f"Evaluated. Model accuracy: {accuracy}")
    logger.info(f"Evaluated. Model accuracy: {precision}")
    logger.info(f"Evaluated. Model accuracy: {recall}")
    logger.info(f"Evaluated. Model accuracy: {_f1_score}")
    logger.info(f"Evaluated. Model accuracy: {_confusion_matrix}")

def save(model):
    logger.info("Saving the model...")

    current_time_stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    model_version = "v" + current_time_stamp

    model_directory = os.path.join('pipeline', 'model')
    os.makedirs(model_directory, exist_ok=True)
    model_path = os.path.join(model_directory, 'model_' + model_version + '.pkl')

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    logger.info(f"Model saved to directory {model_directory}")

def main() -> None:
    """Main script to be executed."""
    data = load()
    data = preprocess(data)

    X_train, X_test, y_train, y_test = split(data)
    model = train(X_train, y_train)
    evaluate(model, X_test, y_test)
    save(model)

if __name__ == '__main__':
    main()