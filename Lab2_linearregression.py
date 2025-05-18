import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load dataset
data = pd.read_csv("dataset.csv")

x = data["x"].values
y = data["y"].values

# Step 2: Calculate means
x_mean = np.mean(x)
y_mean = np.mean(y)

# Step 3: Calculate slope (m) and intercept (b)
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
m = numerator / denominator
b = y_mean - m * x_mean

# Step 4: Predict y values
y_pred = m * x + b

# Print results
print(f"Slope (m): {m:.2f}")
print(f"Intercept (b): {b:.2f}")

# Step 5: Plot results
plt.scatter(x, y, color='blue', label="Actual Data")
plt.plot(x, y_pred, color='red', label="Regression Line")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.legend()
plt.title("Simple Linear Regression using Mean-Based Method")
plt.show()
