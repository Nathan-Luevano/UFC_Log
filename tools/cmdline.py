import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fighter_manager import FighterDB
from core.web_scraper import WebGrabber
from core.predictions import FightPredictor

class CommandLine:
    def __init__(self):
        self.db = FighterDB("data/fighters")
        self.predictor = FightPredictor()
        self.scraper = WebGrabber(workers=4)
        
    def predict_fight(self, fighter1_name: str, fighter2_name: str):
        fighter1 = self.db.find_fighter(fighter1_name)
        fighter2 = self.db.find_fighter(fighter2_name)
        
        if not fighter1:
            print(f"Can't find '{fighter1_name}'")
            return
        if not fighter2:
            print(f"Can't find '{fighter2_name}'")
            return
            
        stats1 = self.db.get_fighter_stats(fighter1)
        stats2 = self.db.get_fighter_stats(fighter2)
        result = self.predictor.predict_fight(stats1, stats2)
        
        print(f"\n{fighter1.name} vs {fighter2.name}")
        print(f"Winner: {result['winner']}")
        print(f"Red: {result['red_win_probability']:.1%}")
        print(f"Blue: {result['blue_win_probability']:.1%}")
        
    def update_db(self, limit=None):
        print("Updating database...")
        added = self.db.update_from_scraper(self.scraper, limit=limit)
        print(f"Added {added} fighters")
        
    def list_fighters(self, search=None, limit=20):
        if search:
            fighters = self.db.search_fighters(search)
            print(f"Fighters matching '{search}':")
        else:
            fighters = self.db.get_all_fighters(limit)
            print(f"Top {limit} fighters:")
            
        for fighter in fighters[:limit]:
            print(f"{fighter.name} ({fighter.wins}-{fighter.losses}) - {fighter.weight}lbs")
            
    def show_info(self):
        info = self.db.get_db_info()
        print("Database Info:")
        print(f"  Total fighters: {info['total_fighters']}")
        print(f"  Last updated: {info['last_updated'][:10] if info['last_updated'] != 'Never' else 'Never'}")

def main():
    parser = argparse.ArgumentParser(description="UFC Fight Predictor")
    subparsers = parser.add_subparsers(dest='command')
    
    predict_cmd = subparsers.add_parser('predict')
    predict_cmd.add_argument('--fighter1', required=True)
    predict_cmd.add_argument('--fighter2', required=True)
    
    update_cmd = subparsers.add_parser('update')
    update_cmd.add_argument('--limit', type=int)
    
    list_cmd = subparsers.add_parser('list')
    list_cmd.add_argument('--search')
    list_cmd.add_argument('--limit', type=int, default=20)
    
    subparsers.add_parser('info')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    cmd = CommandLine()
    
    if args.command == 'predict':
        cmd.predict_fight(args.fighter1, args.fighter2)
    elif args.command == 'update':
        cmd.update_db(args.limit)
    elif args.command == 'list':
        cmd.list_fighters(args.search, args.limit)
    elif args.command == 'info':
        cmd.show_info()

if __name__ == "__main__":
    main()