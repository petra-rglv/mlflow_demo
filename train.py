import numpy as np
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    # Load the Data
    data = pd.read_csv('Iris.csv')
    X = data.drop(['Id', 'Species'], axis=1)
    y = data['Species']

    # Train Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, 
                                                        random_state=5)
    # Train model
    rfc = RandomForestClassifier(n_estimators=50)
    rfc.fit(X_train, y_train)
    
    # Accuracy score
    y_pred = rfc.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy score: %.2f"% accuracy)
    # Track the run
    log_param("n_estimators", rfc.n_estimators)
    log_metric("accuracy", accuracy)
    log_model(rfc, "rfc_model")
    
    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)