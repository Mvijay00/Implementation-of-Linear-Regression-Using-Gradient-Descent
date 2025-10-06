# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

- Load the dataset including numerical features (R&D Spend, Administration, Marketing Spend) and categorical feature (State), then separate the target variable (Profit).

- Apply one-hot encoding to convert the State categorical variable into numerical form while avoiding dummy variable trap by dropping the first category.

- Scale both the combined feature set and the target variable using StandardScaler to normalize data for improved convergence during gradient descent.

- Perform gradient descent to optimize the parameters (theta) of the linear regression model by iteratively updating them to minimize squared errors.

- Predict the profit for new startup data by scaling input features, adding the intercept term, calculating the dot product with learned parameters, and inversely transforming predictions back to original scale for interpretation. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: VIJAYARAGHAVAN M
RegisterNumber:  25017872
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]  # add intercept term
    theta = np.zeros((X.shape[1], 1))  # initialize parameters
    
    for _ in range(num_iters):
        predictions = X.dot(theta)  # predictions shape (m,1)
        errors = predictions - y
        gradient = (1/len(X1)) * X.T.dot(errors)  # gradient vector shape (n+1,1)
        theta -= learning_rate * gradient
        
    return theta

# Load data (make sure the CSV contains columns: R&D Spend, Administration, Marketing Spend, State, Profit)
data = pd.read_csv('50_Startups.csv')

# Extract numerical features
numerical_features = data[['R&D Spend', 'Administration', 'Marketing Spend']]

# One-hot encode "State" categorical feature
encoder = OneHotEncoder(drop='first', sparse=False)  # drop first to avoid multicollinearity
state_encoded = encoder.fit_transform(data[['State']])

# Combine numerical features with one-hot encoded states
X = np.hstack((numerical_features.values, state_encoded))

# Target variable
y = data['Profit'].values.reshape(-1, 1)

# Scale features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Train model using gradient descent
theta = linear_regression(X_scaled, y_scaled)

# Example prediction for new data (with state encoded)
# Format: [R&D Spend, Administration, Marketing Spend, State encoded vars...]
# For example, new startup with values and state 'California':
new_startup = np.array([[165349.2, 136897.8, 471784.1, 0, 1]])  # Assuming encoding drops first state, New York-first, California-second, Florida-third

# Scale new startup features
new_scaled = scaler_X.transform(new_startup)

# Add intercept term for prediction
new_scaled_with_bias = np.c_[np.ones(new_scaled.shape[0]), new_scaled]

# Predict scaled profit
prediction_scaled = new_scaled_with_bias.dot(theta)

# Inverse transform to original profit scale
prediction = scaler_y.inverse_transform(prediction_scaled)

print("Predicted profit (scaled):", prediction_scaled)
print(f"Predicted profit (original scale): {prediction}")

# Plotting the scatter plot of scaled features vs scaled target
plt.scatter(X_train, Y_train, color="orange")
# Plotting the regression line based on the model's predictions
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Scatter plot for test data
plt.scatter(X_test, Y_test, color="blue")
# Regression line for test data predictions
plt.plot(X_test, regressor.predict(X_test), color="green")
plt.title('Hours vs Scores (Testing Set)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

```

## Output:


<img width="1305" height="659" alt="Screenshot 2025-10-06 142000" src="https://github.com/user-attachments/assets/2f3c0706-52e7-4e1c-a0d4-899928cad5ae" />
<img width="1096" height="586" alt="Screenshot 2025-10-06 142027" src="https://github.com/user-attachments/assets/88958424-2482-4a06-8192-610b78716888" />




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
