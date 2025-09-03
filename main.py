from datetime import datetime
import sqlite3
import os
import sys
import time

from core.web_scraper import UFCScraper

class UFCApp:
    def __init__(self):
        self.scraper = UFCScraper()
        self.version = "3.0"
        self.db_path = "data/fighters/fighters.db"
        self.terminal_size = self.get_terminal_size()
        
    def get_terminal_size(self):
        try:
            rows, cols = os.popen('stty size', 'r').read().split()
            return int(rows), int(cols)
        except:
            return 24, 80  # fallback
    
    def clear_screen(self):
        os.system('clear')
        # Disable scrollback and clear everything
        sys.stdout.write('\033[3J\033[H\033[2J')
        sys.stdout.flush()
    
    def center_text(self, text, width=None):
        if width is None:
            width = self.terminal_size[1]
        return text.center(width)
        
    def show_banner(self):
        self.clear_screen()
        banner_lines = [
            '██╗   ██╗███████╗ ██████╗    ██████╗ ██████╗ ███████╗██████╗  █████╗ ████████╗ ██████╗ ██████╗ ',
            '██║   ██║██╔════╝██╔════╝    ██╔══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗   ██╔══╝██╔═══██║██╔══██╗',
            '██║   ██║█████╗  ██║         ██████╔╝██████╔╝█████╗  ██║  ██║███████║   ██║   ██║   ██║██████╔╝',
            '██║   ██║██╔══╝  ██║         ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║██╔══██║   ██║   ██║   ██║██╔══██╗',
            '╚██████╔╝██║     ╚██████╗    ██║     ██║  ██║███████╗██████╔╝██║  ██║   ██║    ██████║ ██║  ██║',
            ' ╚═════╝ ╚═╝      ╚═════╝    ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝',
            '',
            '              UFC Fight Predator v1.0'
        ]
        
        for line in banner_lines:
            print('\033[92m' + self.center_text(line) + '\033[0m')
        
    def show_stats(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM fighters')
        total_fighters = cursor.fetchone()[0]
        conn.close()
        
        stats = [
            f"Version      {self.version}",
            f"Fighters     {total_fighters}",
            f"Engine       Requests + BeautifulSoup",
            f"Status       \033[32mReady\033[0m"
        ]
        
        for line in stats:
            key, value = line.split(' ', 1)
            print(f"\033[36m{key.rjust(12)}:\033[0m {value}")
        print()
        
    def show_menu(self):
        menu = """
┌─ What do you wanna do? ────────────────────────────────┐
│ \033[36m[1] Search Fighters\033[0m            \033[33m[3] Update Database\033[0m     │
│ \033[35m[2] Predict Fight\033[0m              \033[31m[4] Quit\033[0m                │  
└────────────────────────────────────────────────────────┘
"""
        print(menu)
        
    def search_fighters(self):
        current_query = ""
        selected_index = 0
        
        while True:
            self.clear_screen()
            
            # Header
            print("\033[95m" + "="*self.terminal_size[1] + "\033[0m")
            print("\033[95m" + self.center_text("FIGHTER SEARCH") + "\033[0m")
            print("\033[95m" + "="*self.terminal_size[1] + "\033[0m")
            print()
            
            # Search box
            search_box = f"╔ Search: {current_query}║" + " " * (self.terminal_size[1] - len(f"Search: {current_query}") - 10) + "╗"
            print("\033[96m" + search_box + "\033[0m")
            print("\033[90m" + self.center_text("(Type to search, Enter number to select, 'q' to quit)") + "\033[0m")
            print()
            
            # Results
            if current_query:
                matches = self.search_fighters_db(current_query)
                if matches:
                    print(f"\033[92m" + self.center_text(f"Found {len(matches)} fighters") + "\033[0m")
                    print()
                    
                    for i, fighter in enumerate(matches[:12], 1):
                        bg_color = "\033[47m\033[30m" if i == selected_index + 1 else ""
                        reset = "\033[0m" if i == selected_index + 1 else ""
                        
                        name_part = f"{bg_color}[{i:2}] {fighter[4]:<30}{reset}"
                        nickname_part = f" {bg_color}\"{fighter[2]}\"{reset}" if fighter[2] else ""
                        
                        line = f"  {name_part}{nickname_part}"
                        print(line)
                        
                        # Show record if available
                        stats = self.get_fighter_stats_by_id(fighter[0])
                        if stats:
                            record = f"{stats[6] or 0}-{stats[7] or 0}-{stats[8] or 0}"
                            print(f"      \033[90mRecord: {record}\033[0m")
                        print()
                else:
                    print(f"\033[91m" + self.center_text(f"No fighters found matching '{current_query}'") + "\033[0m")
            else:
                recent = self.get_recent_fighters(8)
                if recent:
                    print("\033[92m" + self.center_text("Recent Fighters") + "\033[0m")
                    print()
                    
                    for i, fighter in enumerate(recent, 1):
                        bg_color = "\033[47m\033[30m" if i == selected_index + 1 else ""
                        reset = "\033[0m" if i == selected_index + 1 else ""
                        
                        name_part = f"{bg_color}[{i:2}] {fighter[4]:<30}{reset}"
                        nickname_part = f" {bg_color}\"{fighter[2]}\"{reset}" if fighter[2] else ""
                        
                        line = f"  {name_part}{nickname_part}"
                        print(line)
                        print()
            
            # Input prompt
            print("\n" + "\033[96m" + "═" * self.terminal_size[1] + "\033[0m")
            user_input = input("\033[97m❯ \033[0m").strip()
            
            if user_input.lower() == 'q':
                break
            elif user_input.isdigit():
                idx = int(user_input) - 1
                if current_query:
                    matches = self.search_fighters_db(current_query)
                    if 0 <= idx < len(matches[:12]):
                        self.show_fighter_details(matches[idx])
                        break
                else:
                    recent = self.get_recent_fighters(8)
                    if 0 <= idx < len(recent):
                        self.show_fighter_details(recent[idx])
                        break
            else:
                current_query = user_input
                selected_index = 0
                
    def show_fighter_details(self, fighter_data):
        self.clear_screen()
        
        # fighter_data is tuple: (id, first, last, nickname, full_name, url, scraped)
        fighter_id, first_name, last_name, nickname, full_name, url, _ = fighter_data
        
        # Get stats from fighter_stats table
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM fighter_stats WHERE fighter_id = ?', (fighter_id,))
        stats = cursor.fetchone()
        conn.close()
        
        # Header
        title = full_name
        if nickname:
            title += f' \"{nickname}\"'
            
        print("\033[95m" + "╔" + "═" * (self.terminal_size[1] - 2) + "╗" + "\033[0m")
        print("\033[95m║" + self.center_text(title, self.terminal_size[1] - 2) + "║\033[0m")
        print("\033[95m" + "╠" + "═" * (self.terminal_size[1] - 2) + "╣" + "\033[0m")
        print()
        
        # Basic Info Section
        info_lines = []
        info_lines.append(f"\033[96mFirst Name:\033[0m {first_name or 'N/A'}")
        info_lines.append(f"\033[96mLast Name:\033[0m {last_name or 'N/A'}")
        if nickname:
            info_lines.append(f"\033[96mNickname:\033[0m \"{nickname}\"")
        info_lines.append(f"\033[96mUFC Stats URL:\033[0m \033[94m{url}\033[0m")
        
        for line in info_lines:
            print("  " + line)
        print()
        
        # Stats Section
        if stats:
            # Physical Stats
            print("\033[96m" + self.center_text("═══ PHYSICAL STATS ═══") + "\033[0m")
            print()
            
            physical_stats = [
                f"\033[97mHeight:\033[0m {stats[1] or 'N/A'}",
                f"\033[97mWeight:\033[0m {stats[2] or 'N/A'}",
                f"\033[97mReach:\033[0m {stats[3] or 'N/A'}",
                f"\033[97mStance:\033[0m {stats[4] or 'N/A'}",
                f"\033[97mDate of Birth:\033[0m {stats[5] or 'N/A'}",
            ]
            
            for line in physical_stats:
                print("  " + line)
            print()
            
            # Fight Record
            print("\033[93m" + self.center_text("═══ FIGHT RECORD ═══") + "\033[0m")
            print()
            
            wins = stats[6] if stats[6] is not None else 'N/A'
            losses = stats[7] if stats[7] is not None else 'N/A'
            draws = stats[8] if stats[8] is not None else 'N/A'
            
            record_stats = [
                f"\033[97mWins:\033[0m \033[92m{wins}\033[0m",
                f"\033[97mLosses:\033[0m \033[91m{losses}\033[0m",
                f"\033[97mDraws:\033[0m \033[93m{draws}\033[0m",
            ]
            
            if all(x is not None for x in [stats[6], stats[7], stats[8]]):
                record = f"{stats[6]}-{stats[7]}-{stats[8]}"
                record_stats.append(f"\033[97mRecord:\033[0m \033[1m{record}\033[0m")
            
            for line in record_stats:
                print("  " + line)
            print()
            
            # Career Statistics
            print("\033[92m" + self.center_text("═══ CAREER STATISTICS ═══") + "\033[0m")
            print()
            
            # Striking Stats
            print("  \033[95mSTRIKING:\033[0m")
            slpm = f"{stats[9]:.2f}" if stats[9] is not None else 'N/A'
            str_acc = f"{stats[10]*100:.1f}%" if stats[10] is not None else 'N/A'
            sapm = f"{stats[11]:.2f}" if stats[11] is not None else 'N/A'
            str_def = f"{stats[12]*100:.1f}%" if stats[12] is not None else 'N/A'
            
            striking_stats = [
                f"    \033[97mSLpM (Strikes Landed per Min):\033[0m {slpm}",
                f"    \033[97mStr. Acc. (Striking Accuracy):\033[0m {str_acc}",
                f"    \033[97mSApM (Strikes Absorbed per Min):\033[0m {sapm}",
                f"    \033[97mStr. Def. (Strike Defense):\033[0m {str_def}",
            ]
            
            for line in striking_stats:
                print(line)
            print()
            
            # Grappling Stats
            print("  \033[94mGRAPPLING:\033[0m")
            td_avg = f"{stats[13]:.2f}" if stats[13] is not None else 'N/A'
            td_acc = f"{stats[14]*100:.1f}%" if stats[14] is not None else 'N/A'
            td_def = f"{stats[15]*100:.1f}%" if stats[15] is not None else 'N/A'
            sub_avg = f"{stats[16]:.2f}" if stats[16] is not None else 'N/A'
            
            grappling_stats = [
                f"    \033[97mTD Avg. (Takedowns per Fight):\033[0m {td_avg}",
                f"    \033[97mTD Acc. (Takedown Accuracy):\033[0m {td_acc}",
                f"    \033[97mTD Def. (Takedown Defense):\033[0m {td_def}",
                f"    \033[97mSub. Avg. (Submissions per Fight):\033[0m {sub_avg}",
            ]
            
            for line in grappling_stats:
                print(line)
        else:
            print("\033[91m" + self.center_text("No detailed stats scraped yet") + "\033[0m")
            print("\033[90m" + self.center_text("Run update to get full stats") + "\033[0m")
        
        print("\n\033[95m" + "╚" + "═" * (self.terminal_size[1] - 2) + "╝" + "\033[0m")
        input("\033[90m" + self.center_text("Press Enter to continue...") + "\033[0m")
        
    def predict_fight(self):
        print("\033[2J\033[H")  # Clear screen
        print("\033[91m" + "="*60 + "\033[0m")
        print("\033[91m" + "FIGHT PREDICTION".center(60) + "\033[0m")
        print("\033[91m" + "="*60 + "\033[0m")
        print()
        
        fighter1 = self.get_fighter_for_prediction("\033[31mRed Corner\033[0m: ")
        if not fighter1:
            return
            
        fighter2 = self.get_fighter_for_prediction("\033[34mBlue Corner\033[0m: ")
        if not fighter2:
            return
        
        # Simple prediction based on record
        f1_stats = self.get_fighter_stats_by_id(fighter1[0])
        f2_stats = self.get_fighter_stats_by_id(fighter2[0])
        
        if f1_stats and f2_stats:
            f1_wins = f1_stats[6] or 0
            f1_losses = f1_stats[7] or 0
            f2_wins = f2_stats[6] or 0
            f2_losses = f2_stats[7] or 0
            
            f1_ratio = f1_wins / max(f1_wins + f1_losses, 1)
            f2_ratio = f2_wins / max(f2_wins + f2_losses, 1)
            
            total = f1_ratio + f2_ratio
            if total > 0:
                f1_prob = f1_ratio / total
                f2_prob = f2_ratio / total
            else:
                f1_prob = f2_prob = 0.5
        else:
            f1_prob = f2_prob = 0.5
        
        winner = fighter1[4] if f1_prob > f2_prob else fighter2[4]
        
        print("\033[2J\033[H")
        print("\033[91m" + "═" * 60 + "\033[0m")
        print("\033[91m" + "FIGHT PREDICTION RESULTS".center(60) + "\033[0m")
        print("\033[91m" + "═" * 60 + "\033[0m")
        print()
        print(f"\033[31mRED:\033[0m  {fighter1[4]:<30} ({f1_wins}-{f1_losses})")
        print(f"\033[34mBLUE:\033[0m {fighter2[4]:<30} ({f2_wins}-{f2_losses})")
        print()
        print(f"\033[92mPREDICTED WINNER:\033[0m \033[93m{winner}\033[0m")
        print()
        print(f"\033[97mCHANCES:\033[0m")
        print(f"  Red:   \033[31m{f1_prob:.1%}\033[0m")
        print(f"  Blue:  \033[34m{f2_prob:.1%}\033[0m")
        
        prob_diff = abs(f1_prob - f2_prob)
        if prob_diff > 0.3:
            confidence = "\033[92mPretty confident\033[0m"
        elif prob_diff > 0.15:
            confidence = "\033[93mSomewhat sure\033[0m" 
        else:
            confidence = "\033[91mCould go either way\033[0m"
            
        print(f"\033[97mCONFIDENCE:\033[0m {confidence}")
        print("\n\033[90mPress Enter to continue...\033[0m")
        input()
        
    def get_fighter_for_prediction(self, prompt: str):
        while True:
            name = input(prompt).strip()
            if not name:
                return None
                
            matches = self.search_fighters_db(name)
            if matches:
                if len(matches) == 1:
                    fighter = matches[0]
                    stats = self.get_fighter_stats_by_id(fighter[0])
                    if stats:
                        wins, losses = stats[6] or 0, stats[7] or 0
                        print(f"   \033[92mSelected:\033[0m {fighter[4]} ({wins}-{losses})")
                    else:
                        print(f"   \033[92mSelected:\033[0m {fighter[4]} (No stats)")
                    return fighter
                else:
                    print("   \033[93mMultiple matches:\033[0m")
                    for i, match in enumerate(matches[:5], 1):
                        stats = self.get_fighter_stats_by_id(match[0])
                        if stats:
                            record = f"({stats[6] or 0}-{stats[7] or 0})"
                        else:
                            record = "(No stats)"
                        print(f"     \033[96m{i}.\033[0m {match[4]} {record}")
                    
                    choice = input("   Pick number (or 'n' to try again): ").strip()
                    if choice.isdigit() and 1 <= int(choice) <= len(matches):
                        chosen = matches[int(choice) - 1]
                        stats = self.get_fighter_stats_by_id(chosen[0])
                        if stats:
                            record = f"({stats[6] or 0}-{stats[7] or 0})"
                        else:
                            record = "(No stats)"
                        print(f"   \033[92mSelected:\033[0m {chosen[4]} {record}")
                        return chosen
                    elif choice.lower() == 'n':
                        continue
                    else:
                        return None
            else:
                print(f"   \033[91mCan't find '{name}'\033[0m")
                retry = input("   Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    return None
                        
            
    def update_database(self):
        self.clear_screen()
        
        # Get current stats
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM fighters')
            result = cursor.fetchone()
            current = result[0] if result else 0
            cursor.execute('SELECT COUNT(*) FROM fighters WHERE scraped = TRUE')
            result = cursor.fetchone()
            scraped = result[0] if result else 0
            conn.close()
        except:
            current = 0
            scraped = 0
        
        # Show update options
        print("\033[93m" + "╔" + "═" * (self.terminal_size[1] - 2) + "╗" + "\033[0m")
        print("\033[93m║" + self.center_text("DATABASE UPDATE", self.terminal_size[1] - 2) + "║\033[0m")
        print("\033[93m" + "╠" + "═" * (self.terminal_size[1] - 2) + "╣" + "\033[0m")
        print()
        print(f"  \033[97mCurrent fighters:\033[0m {current}")
        print(f"  \033[97mWith full stats:\033[0m {scraped}")
        print()
        print("  \033[96mUpdate options:\033[0m")
        print("  \033[92m[1]\033[0m Quick update (fighters list only)")
        print("  \033[91m[2]\033[0m Full scrape (10 fighters with stats)")  
        print("  \033[91m[3]\033[0m Full scrape (ALL fighters - takes hours)")
        print("  \033[90m[4]\033[0m Cancel")
        print()
        
        choice = input("  \033[97mChoose option: \033[0m").strip()
        
        if choice == '1':
            self.run_update_with_progress("Quick Update", lambda: self.scraper.scrape_fighters_list_only())
        elif choice == '2':
            self.run_update_with_progress("Full Scrape (10 fighters)", lambda: self.scraper.scrape_all_fighters(self.progress_callback, 10))
        elif choice == '3':
            confirm = input("\n  \033[91mFull scrape of ALL fighters takes hours. Continue? (y/n): \033[0m").strip().lower()
            if confirm == 'y':
                self.run_update_with_progress("Full Scrape (ALL fighters)", lambda: self.scraper.scrape_all_fighters(self.progress_callback))
        
        # Return to home screen - no input needed
        return
        
    def run_update_with_progress(self, operation_name, operation):
        self.clear_screen()
        
        # Header
        print("\033[93m" + "╔" + "═" * (self.terminal_size[1] - 2) + "╗" + "\033[0m")
        print("\033[93m║" + self.center_text(operation_name.upper(), self.terminal_size[1] - 2) + "║\033[0m")
        print("\033[93m" + "╠" + "═" * (self.terminal_size[1] - 2) + "╣" + "\033[0m")
        print()
        
        try:
            operation()
        except Exception as e:
            print(f"\n  \033[91mError: {e}\033[0m")
        
        print(f"\n\033[93m" + "╚" + "═" * (self.terminal_size[1] - 2) + "╝" + "\033[0m")
        print("\n\033[92m" + self.center_text("Update completed! Returning to home screen...") + "\033[0m")
        time.sleep(2)  # Brief pause before returning
        
    def progress_callback(self, message, current, total):
        # Update progress display
        if total > 0:
            percentage = int((current / total) * 100)
            bar_width = min(50, self.terminal_size[1] - 20)
            filled = int((current / total) * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            progress_line = f"  [{bar}] {percentage:3}% ({current}/{total})"
            print(progress_line)
        
        print(f"  \033[96m{message}\033[0m")
        
        # Keep recent messages visible
        sys.stdout.flush()
            
            
        
    def search_fighters_db(self, query):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM fighters 
            WHERE full_name LIKE ? OR first_name LIKE ? OR last_name LIKE ? OR nickname LIKE ?
            ORDER BY full_name
        ''', (f'%{query}%', f'%{query}%', f'%{query}%', f'%{query}%'))
        results = cursor.fetchall()
        conn.close()
        return results
    
    def get_recent_fighters(self, limit=10):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM fighters ORDER BY id DESC LIMIT ?', (limit,))
        results = cursor.fetchall()
        conn.close()
        return results
    
    def get_fighter_stats_by_id(self, fighter_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM fighter_stats WHERE fighter_id = ?', (fighter_id,))
        result = cursor.fetchone()
        conn.close()
        return result
    
    def run(self):
        try:
            print("\033[2J\033[H")  # Clear screen initially
            self.show_banner()
            self.show_stats()
            
            while True:
                self.show_menu()
                
                try:
                    choice = input("Pick: ").strip()
                    
                    if choice == '1':
                        self.search_fighters()
                    elif choice == '2':
                        self.predict_fight()
                    elif choice == '3':
                        self.update_database()
                    elif choice == '4':
                        print("\n\033[92mLater!\033[0m")
                        break
                    else:
                        print("\033[91mPick 1-4\033[0m")
                        
                    if choice != '4':
                        self.show_banner()
                        self.show_stats()
                    
                except KeyboardInterrupt:
                    print("\n\nBye!")
                    break
                except Exception as e:
                    print(f"\nOops: {e}")
                    input("Hit enter to continue...")
                    
        except KeyboardInterrupt:
            print("\n\nSee ya!")
            
if __name__ == "__main__":
    app = UFCApp()
    app.run()