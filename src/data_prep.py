import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def load_and_preprocess_data(file_path= '../data/raw/creditcard.csv', test_size=0.2, random_state=42):
    df = pd.read_csv(file_path)
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    # Make copies of the training and testing sets to avoid SettingWithCopyWarning
    X_train = X_train.copy()
    X_test = X_test.copy()

    scaler = RobustScaler()
    X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
    X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])
    return X_train, X_test, y_train, y_test

