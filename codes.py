import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Titanic dataset
titanic_df = pd.read_csv("train.csv")

# Display the first few rows of the dataset
print(titanic_df.head())

# Preprocessing: Handle missing values and encode categorical variables
titanic_df['Age'].fillna(titanic_df['Age'].median(), inplace=True)
titanic_df['Fare'].fillna(titanic_df['Fare'].median(), inplace=True)

# Encode categorical variables
titanic_df['Sex'] = titanic_df['Sex'].map({'male': 0, 'female': 1})
titanic_df['Pclass'] = titanic_df['Pclass'].astype('category')

# Extract features and target
X = titanic_df[['Sex', 'Pclass', 'Age', 'Fare']]
X = np.c_[np.ones(X.shape[0]), X]  # Add intercept
y = titanic_df['Survived']

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Iteratively Reweighted Least Squares (IRLS) Algorithm
def irls_logistic_regression(X, y, max_iter=100, tol=1e-6):
    beta = np.zeros(X.shape[1])  # Initialize coefficients
    for i in range(max_iter):
        mu = sigmoid(np.dot(X, beta))  # Predicted probabilities
        W = np.diag(mu * (1 - mu))  # Weight matrix
        z = np.dot(X, beta) + (y - mu) / (mu * (1 - mu))  # Working response
        Xt_W_X_inv = np.linalg.inv(np.dot(np.dot(X.T, W), X))
        beta_new = np.dot(np.dot(Xt_W_X_inv, X.T), np.dot(W, z))
        if np.linalg.norm(beta_new - beta) < tol:  # Convergence check
            print(f'Converged in {i+1} iterations.')
            break
        beta = beta_new
    return beta

# Fit the logistic regression model
beta = irls_logistic_regression(X, y)
print("Estimated coefficients:", beta)

# Predictions and accuracy
predictions = sigmoid(np.dot(X, beta))
predicted_classes = (predictions >= 0.5).astype(int)
accuracy = np.mean(predicted_classes == y)
print(f'Accuracy: {accuracy:.4f}')

# Plot Logistic Regression Coefficients
coefficients = ['Intercept', 'Sex (Female)', 'Pclass', 'Age', 'Fare']
beta_values = beta

plt.figure(figsize=(8, 6))
plt.barh(coefficients, beta_values, color='skyblue')
plt.axvline(0, color='gray', linestyle='--')
plt.title('Logistic Regression Coefficients (IRLS)')
plt.xlabel('Coefficient Value')
plt.ylabel('Predictor')
plt.show()

# Example passenger profiles
profiles = [
    {'Sex': 1, 'Pclass': 1, 'Age': 25, 'Fare': 100},
    {'Sex': 0, 'Pclass': 3, 'Age': 40, 'Fare': 10},
    {'Sex': 1, 'Pclass': 2, 'Age': 30, 'Fare': 50},
    {'Sex': 0, 'Pclass': 1, 'Age': 70, 'Fare': 200},
]

profiles_df = pd.DataFrame(profiles)
profiles_df = np.c_[np.ones(profiles_df.shape[0]), profiles_df]  # Add intercept
predicted_probs = sigmoid(np.dot(profiles_df, beta))

# Plot predicted survival probabilities for profiles
profile_labels = ['Female, 1st, Age 25, Fare 100', 'Male, 3rd, Age 40, Fare 10',
                  'Female, 2nd, Age 30, Fare 50', 'Male, 1st, Age 70, Fare 200']

plt.figure(figsize=(8, 6))
plt.bar(profile_labels, predicted_probs, color='lightgreen')
plt.title('Predicted Survival Probabilities by Profile')
plt.ylabel('Survival Probability')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
plt.show()

# Monte Carlo Simulation
np.random.seed(42)
n_simulations = 10000

# Generate synthetic data for Monte Carlo
genders = np.random.choice([0, 1], size=n_simulations, p=[0.5, 0.5])
classes = np.random.choice([1, 2, 3], size=n_simulations, p=[0.2, 0.3, 0.5])
ages = np.random.normal(30, 10, size=n_simulations)
fares = np.random.exponential(50, size=n_simulations)

X = np.column_stack((np.ones(n_simulations), genders, classes, ages, fares))
survival_probs = sigmoid(np.dot(X, beta))
survival_outcomes = np.random.binomial(1, survival_probs)
baseline_survival_prob = np.mean(survival_outcomes)

print(f"Baseline Simulated Survival Probability: {baseline_survival_prob:.4f}")

# Visualization of survival probabilities under scenarios
scenarios = ['Baseline']
probabilities = [baseline_survival_prob]

plt.figure(figsize=(8, 6))
plt.bar(scenarios, probabilities, color='blue')
plt.ylabel('Survival Probability')
plt.title('Survival Probabilities Under Different Scenarios')
plt.ylim(0, 1)
plt.show()
