import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

df = pd.read_csv('clean_ufc_data.csv')

# print(f"Dataset shape: {df.shape}")
# print(f"Missing values:\n{df.isnull().sum()}")

# many missing value del
df_clean = df.dropna().copy()
# print(f"After dropping missing values: {df_clean.shape}")

# composite scores for r &B
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

#scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# accuracy = accuracy_score(y_test, y_pred)
# print(f"\nModel Accuracy: {accuracy:.3f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))
# print("\nConfusion Matrix:")
# print(confusion_matrix(y_test, y_pred))

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'coefficient': model.coef_[0]
})
feature_importance['abs_coefficient'] = abs(feature_importance['coefficient'])
feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

# print("\nTop 10 Most Important Features:")
# print(feature_importance.head(10))

def vis_modl():
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

    print(f"\nModel training complete!")
    print(f"Visualizations saved as 'ufc_model_analysis.png'")
    print(f"Predictions saved as 'model_predictions.csv'")

def predict_fight(red_fighter_stats, blue_fighter_stats):    
    fight_data = {
        'is_title_bout': 0,  
        'weight_class_encoded': 0,  
        'gender_encoded': 0,  
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
    fight_scaled = scaler.transform(fight_df[feature_columns])
    
    prediction = model.predict(fight_scaled)[0]
    probability = model.predict_proba(fight_scaled)[0][1]
    
    return {
        'winner': 'Red Fighter' if prediction == 1 else 'Blue Fighter',
        'red_win_probability': probability,
        'blue_win_probability': 1 - probability
    }

#test
red_fighter = {'wins_total': 16, 'losses_total': 4, 'age': 30, 'height': 75, 'weight': 185, 'reach': 75, 'stance': 'Orthodox', 'SLpM_total': 4.45, 'SApM_total': 3.26, 'sig_str_acc_total': 0.55, 'td_acc_total': 0.32, 'str_def_total': 0.58, 'td_def_total': 0.78, 'sub_avg': 1.1, 'td_avg': 0.85}

blue_fighter = {'wins_total': 17, 'losses_total': 1, 'age': 32, 'height': 73, 'weight': 185, 'reach': 75, 'stance': 'Southpaw', 'SLpM_total': 3.61, 'SApM_total': 2.34, 'sig_str_acc_total': 0.6, 'td_acc_total': 0.6, 'str_def_total': 0.62, 'td_def_total': 0.76, 'sub_avg': 0.5, 'td_avg': 1.56}

prediction_result = predict_fight(red_fighter, blue_fighter)

print(f"Red Fighter: {red_fighter['wins_total']}-{red_fighter['losses_total']}, Age {red_fighter['age']}")
print(f"Blue Fighter: {blue_fighter['wins_total']}-{blue_fighter['losses_total']}, Age {blue_fighter['age']}")
print(f"\nPrediction: {prediction_result['winner']}")
print(f"Red Fighter Win Probability: {prediction_result['red_win_probability']:.2%}")
print(f"Blue Fighter Win Probability: {prediction_result['blue_win_probability']:.2%}")