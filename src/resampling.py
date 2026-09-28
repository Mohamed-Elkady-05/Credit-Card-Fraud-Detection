from imblearn.over_sampling import SMOTE, ADASYN


def apply_smote(X_train, y_train, sampling_strategy='auto', random_state=42):

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    return X_res, y_res