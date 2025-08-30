import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

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

df = pd.read_csv('clean_ufc_data.csv')

print(f"Dataset shape: {df.shape}")
# print(f"Missing values:\n{df.isnull().sum()}")

df_clean = df.dropna().copy()
print(f"After dropping missing values: {df_clean.shape}")

red_score = (
    df_clean['r_wins_total'] * 0.3 + 
    (df_clean['r_wins_total'] / (df_clean['r_wins_total'] + df_clean['r_losses_total'] + 1)) * 0.2 +
    df_clean['r_SLpM_total'] * 0.1 +
    df_clean['r_sig_str_acc_total'] * 0.1 +
    df_clean['r_str_def_total'] * 0.1 +
    (40 - df_clean['r_age']) * 0.05 +  
    (df_clean['r_reach'] / 100) * 0.05 +
    (df_clean['r_height'] / 100) * 0.1
)

blue_score = (
    df_clean['b_wins_total'] * 0.3 + 
    (df_clean['b_wins_total'] / (df_clean['b_wins_total'] + df_clean['b_losses_total'] + 1)) * 0.2 +
    df_clean['b_SLpM_total'] * 0.1 +
    df_clean['b_sig_str_acc_total'] * 0.1 +
    df_clean['b_str_def_total'] * 0.1 +
    (40 - df_clean['b_age']) * 0.05 +
    (df_clean['b_reach'] / 100) * 0.05 +
    (df_clean['b_height'] / 100) * 0.1
)

df_clean['target'] = (red_score > blue_score).astype(int)

# encode categorical variables
le_weight = LabelEncoder()
le_gender = LabelEncoder()
le_r_stance = LabelEncoder()
le_b_stance = LabelEncoder()

df_clean['weight_class_encoded'] = le_weight.fit_transform(df_clean['weight_class'])
df_clean['gender_encoded'] = le_gender.fit_transform(df_clean['gender'])
df_clean['r_stance_encoded'] = le_r_stance.fit_transform(df_clean['r_stance'].fillna('Unknown'))
df_clean['b_stance_encoded'] = le_b_stance.fit_transform(df_clean['b_stance'].fillna('Unknown'))

feature_columns = [
"is_title_bout","weight_class_encoded","gender_encoded",
"r_wins_total","r_losses_total","r_age","r_height","r_weight","r_reach","r_stance_encoded","r_SLpM_total","r_SApM_total","r_sig_str_acc_total","r_td_acc_total","r_str_def_total","r_td_def_total","r_sub_avg","r_td_avg",
"b_wins_total","b_losses_total","b_age","b_height","b_weight","b_reach","b_stance_encoded","b_SLpM_total","b_SApM_total","b_sig_str_acc_total","b_td_acc_total","b_str_def_total","b_td_def_total","b_sub_avg","b_td_avg"
]

X = df_clean[feature_columns]
y = df_clean['target']

scaler = CustomStandardScaler()
X_scaled = scaler.fit_transform(X.values)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=42)

model = CustomLogisticRegression(
    learning_rate=0.05,      
    max_iterations=5000,     
    tolerance=1e-8,          
    l1_reg=0.001,           
    l2_reg=0.01,            
    lr_decay=0.98,          
    patience=100            
)

model.fit(X_train, y_train)

# make predictions with custom model
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.3f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))
# print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'coefficient': model.coef_[0]
})
# feature_importance['abs_coefficient'] = abs(feature_importance['coefficient'])
# feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

# print("\nTop 10 Most Important Features:")
# print(feature_importance.head(10))

def mdl_vs():
    # simple visualizations
    plt.figure(figsize=(15, 10))

    # plot 1: feature importance
    plt.subplot(2, 3, 1)
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['coefficient'])
    plt.yticks(range(len(top_features)), top_features['feature'], fontsize=8)
    plt.xlabel('Coefficient Value')
    plt.title('Top 10 Feature Importance')
    plt.gca().invert_yaxis()

    # plot 2: prediction probability distribution
    plt.subplot(2, 3, 2)
    plt.hist(y_pred_proba, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Prediction Probability (Red Fighter Wins)')
    plt.ylabel('Count')
    plt.title('Prediction Probability Distribution')

    # plot 3: wins vs losses for red fighters
    plt.subplot(2, 3, 3)
    plt.scatter(df_clean['r_wins_total'], df_clean['r_losses_total'], 
            c=df_clean['target'], alpha=0.6, cmap='coolwarm')
    plt.xlabel('Red Fighter Wins')
    plt.ylabel('Red Fighter Losses')
    plt.title('Red Fighter Record vs Outcome')
    plt.colorbar(label='Red Fighter Won')

    # plot 4: age difference impact
    plt.subplot(2, 3, 4)
    plt.boxplot([df_clean[df_clean['target']==0]['age_diff'], 
                df_clean[df_clean['target']==1]['age_diff']], 
            tick_labels=['Blue Won', 'Red Won'])
    plt.ylabel('Age Difference (Red - Blue)')
    plt.title('Age Difference Impact on Outcome')

    # plot 5: weight class distribution
    plt.subplot(2, 3, 5)
    weight_counts = df_clean['weight_class'].value_counts()
    plt.bar(range(len(weight_counts)), weight_counts.values)
    plt.xticks(range(len(weight_counts)), weight_counts.index, rotation=45, ha='right', fontsize=8)
    plt.ylabel('Number of Fights')
    plt.title('Fights by Weight Class')

    # plot 6: target distribution
    plt.subplot(2, 3, 6)
    target_counts = df_clean['target'].value_counts()
    plt.bar(['Blue Fighter Wins', 'Red Fighter Wins'], target_counts.values)
    plt.ylabel('Number of Fights')
    plt.title('Target Distribution')
    for i, v in enumerate(target_counts.values):
        plt.text(i, v + 50, str(v), ha='center')

    plt.tight_layout()
    plt.savefig('ufc_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

results_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred,
    'probability': y_pred_proba
})
results_df.to_csv('model_predictions.csv', index=False)

def plot_training_curves():
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(model.cost_history, label='Training Cost', alpha=0.7)
    plt.plot(model.val_cost_history, label='Validation Cost', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Training vs Validation Cost')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Plot last 500 iterations for better detail
    start_idx = max(0, len(model.cost_history) - 500)
    plt.plot(model.cost_history[start_idx:], label='Training Cost', alpha=0.7)
    plt.plot(model.val_cost_history[start_idx:], label='Validation Cost', alpha=0.7)
    plt.xlabel('Iteration (Last 500)')
    plt.ylabel('Cost')
    plt.title('Training Curves (Detailed View)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

print(f"Visualizations saved as 'ufc_model_analysis.png'")
print(f"Predictions saved as 'model_predictions.csv'")
print(f"Training curves saved as 'training_curves.png'")

# Plot training curves to check for overfitting
# plot_training_curves()

def predict_fight(red_fighter_stats, blue_fighter_stats):    
    fight_data = {
        'is_title_bout': 0,  
        'weight_class_encoded': 0,  
        'gender_encoded': 0,  # assume male
        'r_wins_total': red_fighter_stats['wins_total'],
        'r_losses_total': red_fighter_stats['losses_total'],
        'r_age': red_fighter_stats['age'],
        'r_height': red_fighter_stats['height'],
        'r_weight': red_fighter_stats['weight'],
        'r_reach': red_fighter_stats['reach'],
        'r_stance_encoded': 0,  
        'r_SLpM_total': red_fighter_stats['SLpM_total'],
        'r_SApM_total': red_fighter_stats['SApM_total'],
        'r_sig_str_acc_total': red_fighter_stats['sig_str_acc_total'],
        'r_td_acc_total': red_fighter_stats['td_acc_total'],
        'r_str_def_total': red_fighter_stats['str_def_total'],
        'r_td_def_total': red_fighter_stats['td_def_total'],
        'r_sub_avg': red_fighter_stats['sub_avg'],
        'r_td_avg': red_fighter_stats['td_avg'],
        'b_wins_total': blue_fighter_stats['wins_total'],
        'b_losses_total': blue_fighter_stats['losses_total'],
        'b_age': blue_fighter_stats['age'],
        'b_height': blue_fighter_stats['height'],
        'b_weight': blue_fighter_stats['weight'],
        'b_reach': blue_fighter_stats['reach'],
        'b_stance_encoded': 0,  
        'b_SLpM_total': blue_fighter_stats['SLpM_total'],
        'b_SApM_total': blue_fighter_stats['SApM_total'],
        'b_sig_str_acc_total': blue_fighter_stats['sig_str_acc_total'],
        'b_td_acc_total': blue_fighter_stats['td_acc_total'],
        'b_str_def_total': blue_fighter_stats['str_def_total'],
        'b_td_def_total': blue_fighter_stats['td_def_total'],
        'b_sub_avg': blue_fighter_stats['sub_avg'],
        'b_td_avg': blue_fighter_stats['td_avg']
    }
    
    # Encode stances
    red_stance = red_fighter_stats.get('stance', 'Unknown')
    blue_stance = blue_fighter_stats.get('stance', 'Unknown')
    
    try:
        fight_data['r_stance_encoded'] = le_r_stance.transform([red_stance])[0] if red_stance in le_r_stance.classes_ else 0
    except:
        fight_data['r_stance_encoded'] = 0
        
    try:
        fight_data['b_stance_encoded'] = le_b_stance.transform([blue_stance])[0] if blue_stance in le_b_stance.classes_ else 0
    except:
        fight_data['b_stance_encoded'] = 0
    
    fight_df = pd.DataFrame([fight_data])
    fight_scaled = scaler.transform(fight_df[feature_columns].values)
    
    prediction = model.predict(fight_scaled)[0]
    probability = model.predict_proba(fight_scaled)[0][1]
    
    return {
        'winner': 'Red Fighter' if prediction == 1 else 'Blue Fighter',
        'red_win_probability': probability,
        'blue_win_probability': 1 - probability
    }

red_fighter = {'wins_total': 18, 'losses_total': 2, 'age': 28, 'height': 71, 'weight': 170, 'reach': 73, 'stance': 'Switch', 'SLpM_total': 6.84, 'SApM_total': 4.53, 'sig_str_acc_total': 0.52, 'td_acc_total': 0.11, 'str_def_total': 0.64, 'td_def_total': 0.69, 'sub_avg': 0.2, 'td_avg': 0.16}

blue_fighter = {'wins_total': 24, 'losses_total': 4, 'age': 37, 'height': 71, 'weight': 170, 'reach': 72, 'stance': 'Orthodox', 'SLpM_total': 4.46, 'SApM_total': 3.78, 'sig_str_acc_total': 0.43, 'td_acc_total': 0.38, 'str_def_total': 0.55, 'td_def_total': 0.9, 'sub_avg': 0.1, 'td_avg': 2.24}

prediction_result = predict_fight(red_fighter, blue_fighter)

print(f"Red Fighter: {red_fighter['wins_total']}-{red_fighter['losses_total']}, Age {red_fighter['age']}")
print(f"Blue Fighter: {blue_fighter['wins_total']}-{blue_fighter['losses_total']}, Age {blue_fighter['age']}")
print(f"\nPrediction: {prediction_result['winner']}")
print(f"Red Fighter Win Probability: {prediction_result['red_win_probability']:.1%}")
print(f"Blue Fighter Win Probability: {prediction_result['blue_win_probability']:.1%}")