# Titanic-survival-prediction-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load datasets
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Select features
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

# Handle missing values
train_df["Age"].fillna(train_df["Age"].median(), inplace=True)
test_df["Age"].fillna(test_df["Age"].median(), inplace=True)
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

# Encode categorical data
train_df["Sex"] = train_df["Sex"].map({"male": 0, "female": 1})
test_df["Sex"] = test_df["Sex"].map({"male": 0, "female": 1})

# Prepare training data
X_train = train_df[features]
y_train = train_df["Survived"]
X_test = test_df[features]

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": predictions
})

submission.to_csv("output/submission.csv", index=False)

print("✅ submission.csv generated successfully!")
