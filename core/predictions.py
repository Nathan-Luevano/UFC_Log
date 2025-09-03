import json
import os

class FightPredictor:
    def __init__(self):
        self.trained = False
        self.weights = {
            'wins_importance': 0.3,
            'age_penalty': -0.1,
            'height_bonus': 0.05,
            'reach_bonus': 0.1,
            'experience_bonus': 0.15
        }
        
    def load_model(self, model_path: str) -> bool:
        if os.path.exists(model_path):
            try:
                with open(model_path, 'r') as f:
                    data = json.load(f)
                self.weights = data.get('weights', self.weights)
                self.trained = True
                return True
            except:
                pass
        return False
        
    def save_model(self, model_path: str):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        data = {'weights': self.weights, 'trained': self.trained}
        with open(model_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def calc_fighter_score(self, stats: dict) -> float:
        score = 0.0
        
        wins = stats.get('wins_total', 0)
        losses = stats.get('losses_total', 0)
        total_fights = wins + losses
        
        if total_fights > 0:
            win_rate = wins / total_fights
            score += win_rate * self.weights['wins_importance']
            
        age = stats.get('age', 30)
        if age > 0:
            age_factor = max(0, 1 - abs(age - 30) * 0.02)
            score += age_factor * abs(self.weights['age_penalty'])
            
        height = stats.get('height', 70)
        reach = stats.get('reach', 72)
        
        score += (height / 72) * self.weights['height_bonus']
        score += (reach / 74) * self.weights['reach_bonus']
        
        if total_fights > 5:
            experience_factor = min(1.0, total_fights / 20)
            score += experience_factor * self.weights['experience_bonus']
            
        return score
        
    def predict_fight(self, red_stats: dict, blue_stats: dict) -> dict:
        red_score = self.calc_fighter_score(red_stats)
        blue_score = self.calc_fighter_score(blue_stats)
        
        score_diff = red_score - blue_score
        
        if score_diff > 0:
            red_prob = 0.5 + min(0.4, score_diff * 0.3)
        else:
            red_prob = 0.5 + max(-0.4, score_diff * 0.3)
            
        blue_prob = 1.0 - red_prob
        
        winner = "Red Fighter" if red_prob > 0.5 else "Blue Fighter"
        
        return {
            'winner': winner,
            'red_win_probability': red_prob,
            'blue_win_probability': blue_prob
        }

# for compatibility
UFCModel = FightPredictor