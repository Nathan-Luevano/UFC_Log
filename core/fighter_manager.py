import json
import os
import re
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

@dataclass
class Fighter:
    name: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    age: int = 0
    height: int = 0
    weight: int = 0
    reach: int = 0
    stance: str = "Unknown"
    slpm: float = 0.0
    sapm: float = 0.0
    str_acc: float = 0.0
    str_def: float = 0.0
    td_avg: float = 0.0
    td_acc: float = 0.0
    td_def: float = 0.0
    sub_avg: float = 0.0
    last_updated: str = ""
    profile_url: str = ""

class FighterDB:
    def __init__(self, db_path: str = "data/fighter_db"):
        self.db_path = db_path
        self.fighters = {}
        self.make_dir()
        self.load_data()
        
    def make_dir(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
    def clean_name(self, name: str) -> str:
        return re.sub(r'[^\w\s]', '', name.strip().lower())
        
    def str_to_float(self, value: str, is_percent: bool = False) -> float:
        if not value or value == "N/A":
            return 0.0
        try:
            value = re.sub(r'[^\d\.-]', '', str(value))
            if not value:
                return 0.0
            result = float(value)
            if is_percent and result > 1.0:
                result = result / 100.0
            return result
        except (ValueError, TypeError):
            return 0.0
            
    def height_to_inches(self, height_str: str) -> int:
        if not height_str:
            return 0
        match = re.search(r"(\d+)['ft]*\s*(\d+)", height_str)
        if match:
            feet = int(match.group(1))
            inches = int(match.group(2))
            return feet * 12 + inches
        feet_match = re.search(r"(\d+)'", height_str)
        if feet_match:
            return int(feet_match.group(1)) * 12
        return 0
        
    def parse_record(self, record: str) -> Tuple[int, int, int]:
        if not record:
            return 0, 0, 0
        match = re.search(r"(\d+)-(\d+)(?:-(\d+))?", record)
        if match:
            wins = int(match.group(1))
            losses = int(match.group(2))
            draws = int(match.group(3)) if match.group(3) else 0
            return wins, losses, draws
        return 0, 0, 0
        
    def calc_age(self, dob: str) -> int:
        if not dob:
            return 0
        try:
            for fmt in ['%b %d, %Y', '%B %d, %Y', '%m/%d/%Y', '%Y-%m-%d']:
                try:
                    birth_date = datetime.strptime(dob, fmt).date()
                    today = date.today()
                    age = today.year - birth_date.year
                    if (today.month, today.day) < (birth_date.month, birth_date.day):
                        age -= 1
                    return max(0, age)
                except ValueError:
                    continue
        except:
            pass
        return 0
        
    def add_fighter_from_scraped(self, data: Dict) -> Optional[Fighter]:
        if not data.get('name'):
            return None
            
        wins, losses, draws = self.parse_record(data.get('record', ''))
        
        fighter = Fighter(
            name=data['name'],
            wins=wins,
            losses=losses,
            draws=draws,
            height=self.height_to_inches(data.get('height', '')),
            weight=int(re.sub(r'[^\d]', '', data.get('weight', '0')) or 0),
            reach=int(re.sub(r'[^\d]', '', data.get('reach', '0')) or 0),
            stance=data.get('stance', 'Unknown').strip() or 'Unknown',
            age=self.calc_age(data.get('dob', '')),
            slpm=self.str_to_float(data.get('slpm', 0)),
            sapm=self.str_to_float(data.get('sapm', 0)),
            str_acc=self.str_to_float(data.get('str_acc', 0), is_percent=True),
            str_def=self.str_to_float(data.get('str_def', 0), is_percent=True),
            td_avg=self.str_to_float(data.get('td_avg', 0)),
            td_acc=self.str_to_float(data.get('td_acc', 0), is_percent=True),
            td_def=self.str_to_float(data.get('td_def', 0), is_percent=True),
            sub_avg=self.str_to_float(data.get('sub_avg', 0)),
            last_updated=data.get('scraped_date', datetime.now().isoformat()),
            profile_url=data.get('url', '')
        )
        
        clean_name = self.clean_name(fighter.name)
        self.fighters[clean_name] = fighter
        return fighter
        
    def find_fighter(self, name: str) -> Optional[Fighter]:
        clean_name = self.clean_name(name)
        
        if clean_name in self.fighters:
            return self.fighters[clean_name]
            
        for stored_name, fighter in self.fighters.items():
            if clean_name in stored_name or stored_name in clean_name:
                return fighter
                
        clean_words = clean_name.split()
        if len(clean_words) > 1:
            for stored_name, fighter in self.fighters.items():
                stored_words = stored_name.split()
                if any(word in stored_words for word in clean_words):
                    return fighter
                    
        return None
        
    def search_fighters(self, query: str) -> List[Fighter]:
        query_clean = self.clean_name(query)
        matches = []
        
        for fighter in self.fighters.values():
            fighter_name_clean = self.clean_name(fighter.name)
            if query_clean in fighter_name_clean:
                matches.append(fighter)
        
        if len(matches) < 5:
            query_words = query_clean.split()
            for fighter in self.fighters.values():
                if fighter not in matches:
                    fighter_name_clean = self.clean_name(fighter.name)
                    fighter_words = fighter_name_clean.split()
                    if any(word in fighter_words for word in query_words):
                        matches.append(fighter)
                        
        return matches[:20]
        
    def get_fighter_stats(self, fighter: Fighter) -> Dict:
        return {
            'wins_total': fighter.wins,
            'losses_total': fighter.losses,
            'age': fighter.age,
            'height': fighter.height,
            'weight': fighter.weight,
            'reach': fighter.reach,
            'stance': fighter.stance,
            'SLpM_total': fighter.slpm,
            'SApM_total': fighter.sapm,
            'sig_str_acc_total': fighter.str_acc,
            'td_acc_total': fighter.td_acc,
            'str_def_total': fighter.str_def,
            'td_def_total': fighter.td_def,
            'sub_avg': fighter.sub_avg,
            'td_avg': fighter.td_avg
        }
        
    def load_from_scraped_data(self, scraped_data: List[Dict]) -> int:
        added = 0
        for fighter_data in scraped_data:
            fighter = self.add_fighter_from_scraped(fighter_data)
            if fighter:
                added += 1
        print(f"Added {added} fighters")
        return added
        
    def load_from_file(self, filepath: str) -> int:
        try:
            if filepath.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return self.load_from_scraped_data(data)
            else:
                print(f"Can't load {filepath}")
                return 0
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return 0
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return 0
            
    def save_data(self):
        data = []
        for fighter in self.fighters.values():
            fighter_dict = asdict(fighter)
            data.append(fighter_dict)
            
        with open(f"{self.db_path}.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"Saved {len(data)} fighters")
        
    def load_data(self):
        json_path = f"{self.db_path}.json"
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for fighter_data in data:
                    fighter = Fighter(**fighter_data)
                    clean_name = self.clean_name(fighter.name)
                    self.fighters[clean_name] = fighter
                    
                print(f"Loaded {len(self.fighters)} fighters")
            except Exception as e:
                print(f"Error loading: {e}")
                self.fighters = {}
                
    def get_all_fighters(self, limit: int = None) -> List[Fighter]:
        fighters_list = sorted(
            self.fighters.values(), 
            key=lambda f: (f.wins, -f.losses), 
            reverse=True
        )
        if limit:
            return fighters_list[:limit]
        return fighters_list
        
    def get_db_info(self) -> Dict:
        if not self.fighters:
            return {
                'total_fighters': 0,
                'database_path': self.db_path,
                'last_updated': "Never"
            }
            
        last_updated = max(
            (f.last_updated for f in self.fighters.values() if f.last_updated), 
            default="Never"
        )
        
        total_fights = sum(f.wins + f.losses for f in self.fighters.values())
        active_fighters = sum(1 for f in self.fighters.values() if f.wins + f.losses > 0)
        
        return {
            'total_fighters': len(self.fighters),
            'active_fighters': active_fighters,
            'total_fights': total_fights,
            'database_path': self.db_path,
            'last_updated': last_updated
        }
        
    def update_from_scraper(self, scraper, limit: int = None):
        print("Getting fresh fighter data...")
        scraped_data = scraper.get_all_fighters(limit=limit)
        
        if scraped_data:
            added = self.load_from_scraped_data(scraped_data)
            self.save_data()
            return added
        return 0

if __name__ == "__main__":
    db = FighterDB("test_db")
    
    test_fighter = {
        'name': 'Jon Jones',
        'record': '27-1-1',
        'height': "6' 4\"",
        'weight': '205 lbs',
        'reach': '84.5"',
        'stance': 'Orthodox',
        'dob': 'Jul 19, 1987',
        'slpm': '4.29',
        'sapm': '2.05',
        'str_acc': '58%',
        'str_def': '62%',
        'td_avg': '2.07',
        'td_acc': '43%',
        'td_def': '95%',
        'sub_avg': '0.4',
        'scraped_date': datetime.now().isoformat()
    }
    
    fighter = db.add_fighter_from_scraped(test_fighter)
    print(f"Added: {fighter.name} ({fighter.wins}-{fighter.losses}-{fighter.draws})")
    
    found = db.find_fighter("jon jones")
    print(f"Found: {found.name if found else 'Not found'}")
    
    db.save_data()
    print("Done")