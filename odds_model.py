# Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6, 
                 l1_reg=0.0, l2_reg=0.01, lr_decay=0.95, patience=50):
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.l1_reg = l1_reg  
        self.l2_reg = l2_reg  
        self.lr_decay = lr_decay  
        self.patience = patience  
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.val_cost_history = []
        
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, y_true, y_pred, weights):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        m = len(y_true)
        base_cost = -1/m * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        l1_cost = self.l1_reg * np.sum(np.abs(weights))
        l2_cost = self.l2_reg * np.sum(weights ** 2)
        
        total_cost = base_cost + l1_cost + l2_cost
        return total_cost
    
    def fit(self, X, y, X_val=None, y_val=None):
        m, n = X.shape
        self.weights = np.random.normal(0, 0.01, n)
        self.bias = 0
        
        if X_val is None:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
        else:
            X_train, y_train = X, y
        
        m_train = len(y_train)
        best_val_cost = float('inf')
        patience_counter = 0
        
        for i in range(self.max_iterations):
            z_train = np.dot(X_train, self.weights) + self.bias
            y_pred_train = self._sigmoid(z_train)
            
            train_cost = self._compute_cost(y_train, y_pred_train, self.weights)
            self.cost_history.append(train_cost)
            
            z_val = np.dot(X_val, self.weights) + self.bias
            y_pred_val = self._sigmoid(z_val)
            val_cost = self._compute_cost(y_val, y_pred_val, np.zeros_like(self.weights))
            self.val_cost_history.append(val_cost)
            
            dw = (1/m_train) * np.dot(X_train.T, (y_pred_train - y_train))
            dw += self.l1_reg * np.sign(self.weights) + 2 * self.l2_reg * self.weights
            
            db = (1/m_train) * np.sum(y_pred_train - y_train)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if (i + 1) % 100 == 0:
                self.learning_rate = self.learning_rate * self.lr_decay
            
            if val_cost < best_val_cost:
                best_val_cost = val_cost
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"Early stopping at iteration {i+1} (val_cost: {val_cost:.6f})")
                break
            
            # Convergence check on training cost
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Converged after {i+1} iterations (train_cost: {train_cost:.6f})")
                break
        
        print(f"Final train cost: {self.cost_history[-1]:.6f}, val cost: {self.val_cost_history[-1]:.6f}")
        return self
    
    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        probabilities = self._sigmoid(z)
        return np.column_stack([1 - probabilities, probabilities])
    
    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= 0.5).astype(int)
    
    # coefficients (weights)
    @property
    def coef_(self):
        return self.weights.reshape(1, -1) if self.weights is not None else None

class CustomStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None
        
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1
        return self
        
    def transform(self, X):
        return (X - self.mean_) / self.std_
        
    def fit_transform(self, X):
        return self.fit(X).transform(X)



# Input Data
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename)) 


# Load in dataset
df = pd.read_csv("ufc-master.csv")

fighter_names = df[['R_fighter', 'B_fighter']]
odds_all = df[['R_odds', 'B_odds']]



# Define features X and labels y
X = df[['R_odds', 'B_odds']].fillna(0)

df.loc[df['Winner'] == 'Red', 'label'] = 0
df.loc[df['Winner'] == 'Blue', 'label'] = 1

# Convert dtype to int explicitly to avoid floats like 0.0, 1.0
df['label'] = df['label'].astype(int)

y = df['label'].values



# Train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, names_train, names_test, odds_train, odds_test = train_test_split(
    X, y, fighter_names, odds_all, test_size=0.5, random_state=42
)

# Scale features
scaler = CustomStandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train classifier
my_classifier = CustomLogisticRegression()
my_classifier.fit(X_train, y_train)

# Make predictions and probabilities
predictions = my_classifier.predict(X_test)
probs = my_classifier.predict_proba(X_test)


# Evaluate accuracy
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, predictions))


# EV function
def get_bet_ev(odds, prob):
    if odds > 0:  # e.g. +150
        return (odds * prob) - (100 * (1 - prob))
    else:  # e.g. -150
        return ((100 / abs(odds)) * 100 * prob) - (100 * (1 - prob))


# Build results dataframe
results = names_test.copy().reset_index(drop=True)
results['R_odds'] = odds_test.reset_index(drop=True)['R_odds']
results['B_odds'] = odds_test.reset_index(drop=True)['B_odds']
results['P_red'] = probs[:, 0]
results['P_blue'] = probs[:, 1]
results['EV_red'] = results.apply(lambda row: get_bet_ev(row['R_odds'], row['P_red']), axis=1)
results['EV_blue'] = results.apply(lambda row: get_bet_ev(row['B_odds'], row['P_blue']), axis=1)

print("\nTop of results dataframe:")
print(results.head(10))


# Build probs_list for fight-by-fight printout
probs_list = []
for i in range(len(results)):
    fighters = (results.loc[i, 'R_fighter'], results.loc[i, 'B_fighter'])
    odds = (results.loc[i, 'R_odds'], results.loc[i, 'B_odds'])
    probs_tuple = (results.loc[i, 'P_red'], results.loc[i, 'P_blue'])
    probs_list.append((fighters, odds, probs_tuple))


# Pretty print fight-by-fight EV analysis
for p in probs_list:
    red_ev = get_bet_ev(p[1][0], p[2][0])
    blue_ev = get_bet_ev(p[1][1], p[2][1])
    
    print(f"\n{p[0][0]} (RED) vs {p[0][1]} (BLUE)")
    print(f"{p[0][0]} has a {p[2][0]*100:.2f}% chance of winning. Odds: {p[1][0]}. EV: {red_ev:.2f}")
    print(f"{p[0][1]} has a {p[2][1]*100:.2f}% chance of winning. Odds: {p[1][1]}. EV: {blue_ev:.2f}")
    
    if red_ev > 0:
        print("RED is a good bet")
    elif blue_ev > 0:
        print("BLUE is a good bet")
    else:
        print("There is NO good bet")