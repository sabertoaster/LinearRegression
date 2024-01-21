#%%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('winequality-red.csv')
print(df.head())
#predict pH => swap pH and alcohol
data = df.iloc[:, :-1].to_numpy().T
data[[8,10]] = data[[10,8]]
data = data.T

print(f"Predict pH of a sample by using Linear Regression through this dataset, with 10 other features:\n{data}")
np.random.shuffle(data)

# %%

class Model():
    
    def __init__(self, data, lr, epochs):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.33, random_state=42)
        self.lr = lr
        self.epochs = epochs
        self.weight = np.random.randn(self.X_train.shape[1], 1)
        
    def predict(self, X):
        return self.weight.T @ X

    def test_eval(self): 
        y_hat = self.predict(self.X_test.T)
        return np.mean(np.abs(y_hat - self.y_test))
    
    # def gradient(self, y, y_hat, X):
    #     result = (y_hat - y) * X / len(X)
    #     return result.reshape(-1, 1)
    
    def fit(self):
        self.weight = np.linalg.pinv(self.X_train.T @ self.X_train) @ np.dot(self.X_train.T, self.y_train)
                
linreg = Model(data, 0.01, 15)
linreg.fit()
print(f"Mean of MSE in test set {linreg.test_eval()}")
# %%
