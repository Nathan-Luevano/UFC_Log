import requests
import requests.exceptions
from bs4 import BeautifulSoup
import sqlite3
import time
from urllib.parse import urljoin

class UFCScraper:
    def __init__(self, db_path="data/fighters/fighters.db"):
        self.base_url = "http://ufcstats.com"
        self.fighters_urls = []
        for i in 'abcdefghijklmnopqrstuvwxyz':
            self.fighters_urls.append(f"http://ufcstats.com/statistics/fighters?char={i}&page=all")
        self.db_path = db_path
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.setup_database()
    
    def setup_database(self):
        # Ensure directory exists
        import os
        db_dir = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS fighters (
                id INTEGER PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                nickname TEXT,
                full_name TEXT UNIQUE,
                url TEXT,
                scraped BOOLEAN DEFAULT FALSE
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS fighter_stats (
                fighter_id INTEGER,
                height TEXT,
                weight TEXT,
                reach TEXT,
                stance TEXT,
                dob TEXT,
                wins INTEGER,
                losses INTEGER,
                draws INTEGER,
                slpm REAL,
                str_acc REAL,
                sapm REAL,
                str_def REAL,
                td_avg REAL,
                td_acc REAL,
                td_def REAL,
                sub_avg REAL,
                FOREIGN KEY (fighter_id) REFERENCES fighters (id)
            )
        ''')
        conn.commit()
        conn.close()
    
    def get_fighters_list(self, progress_callback=None):
        all_fighters = []
        url_to_names = {}
        
        total_pages = len(self.fighters_urls)
        
        for page_num, fighters_url in enumerate(self.fighters_urls, 1):
            char = fighters_url.split('char=')[1].split('&')[0].upper()
            
            if progress_callback:
                progress_callback(f"Fetching fighters starting with '{char}'...", page_num - 1, total_pages)
            
            try:
                response = self.session.get(fighters_url, timeout=30)
                response.raise_for_status()
            except requests.exceptions.Timeout:
                raise Exception(f"Connection timeout while fetching '{char}' fighters")
            except requests.exceptions.ConnectionError:
                raise Exception("Connection error - check internet connection")
            except requests.exceptions.HTTPError as e:
                raise Exception(f"HTTP error {e.response.status_code} while fetching '{char}' fighters")
            except Exception as e:
                raise Exception(f"Unexpected error fetching '{char}' fighters: {str(e)}")
            
            try:
                soup = BeautifulSoup(response.content, 'html.parser')
                fighter_links = soup.find_all('a', href=True)
                
                page_fighters = 0
                # Group names by URL for this page
                for link in fighter_links:
                    href = link.get('href')
                    if href and '/fighter-details/' in href:
                        name = link.text.strip()
                        if name:
                            full_url = urljoin(self.base_url, href)
                            if full_url not in url_to_names:
                                url_to_names[full_url] = []
                                page_fighters += 1
                            url_to_names[full_url].append(name)
                
                if progress_callback:
                    progress_callback(f"✓ Found {page_fighters} fighters starting with '{char}'", page_num, total_pages)
                
                time.sleep(0.5)  # Rate limiting between pages
                
            except Exception as e:
                if progress_callback:
                    progress_callback(f"✗ Error parsing '{char}' page: {str(e)}", page_num, total_pages)
                continue
        
        # Process all grouped names
        fighters = []
        for url, names in url_to_names.items():
            if len(names) >= 2:
                first_name = names[0]
                last_name = names[1] 
                nickname = names[2] if len(names) >= 3 else None
                full_name = f"{first_name} {last_name}"
                fighters.append((first_name, last_name, nickname, full_name, url))
            elif len(names) == 1:
                # Single name - treat as full name
                full_name = names[0]
                fighters.append((None, None, None, full_name, url))
        
        if not fighters:
            raise Exception("No fighters found across all pages - possible website structure change")
        
        return fighters
    
    def save_fighters_to_db(self, fighters):
        conn = sqlite3.connect(self.db_path)
        conn.executemany(
            'INSERT OR IGNORE INTO fighters (first_name, last_name, nickname, full_name, url) VALUES (?, ?, ?, ?, ?)',
            fighters
        )
        conn.commit()
        conn.close()
    
    def get_fighter_stats(self, fighter_url):
        try:
            response = self.session.get(fighter_url, timeout=30)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            raise Exception("Connection timeout")
        except requests.exceptions.ConnectionError:
            raise Exception("Connection error")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise Exception("Fighter page not found")
            else:
                raise Exception(f"HTTP error {e.response.status_code}")
        except Exception as e:
            raise Exception(f"Request failed: {str(e)}")
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        stats = {
            'height': None, 'weight': None, 'reach': None, 'stance': None, 'dob': None,
            'wins': None, 'losses': None, 'draws': None,
            'slpm': None, 'str_acc': None, 'sapm': None, 'str_def': None,
            'td_avg': None, 'td_acc': None, 'td_def': None, 'sub_avg': None
        }
        
        # Parse record from title section
        record_span = soup.find('span', class_='b-content__title-record')
        if record_span:
            record_text = record_span.get_text(strip=True)
            if 'Record:' in record_text:
                record = record_text.split('Record:')[1].strip()
                parts = record.split('-')
                if len(parts) >= 2:
                    stats['wins'] = int(parts[0]) if parts[0].isdigit() else None
                    stats['losses'] = int(parts[1]) if parts[1].isdigit() else None
                    if len(parts) >= 3 and parts[2].isdigit():
                        stats['draws'] = int(parts[2])
                    else:
                        stats['draws'] = None
        
        # Parse basic physical stats - multiple approaches
        # Method 1: Look for the standard info boxes
        info_boxes = soup.find_all('div', class_='b-list__info-box b-list__info-box_style_small-width js-guide')
        
        for box in info_boxes:
            label_elem = box.find('i', class_='b-list__info-box-label')
            if label_elem:
                label = label_elem.get_text(strip=True).lower()
                value_elem = box.find('i', class_='b-list__info-box-value')
                if value_elem:
                    value = value_elem.get_text(strip=True)
                    
                    if 'height' in label:
                        stats['height'] = value if value != '--' else None
                    elif 'weight' in label:
                        stats['weight'] = value if value != '--' else None
                    elif 'reach' in label:
                        stats['reach'] = value if value != '--' else None
                    elif 'stance' in label:
                        stats['stance'] = value if value != '--' else None
                    elif 'dob' in label:
                        stats['dob'] = value if value != '--' else None
        
        # Method 2: Look for alternative structures
        if not stats['height']:  # Try alternative parsing
            # Look for any list items containing physical stats
            all_list_items = soup.find_all(['li', 'p', 'div'])
            for item in all_list_items:
                text = item.get_text(strip=True)
                if ':' in text:
                    parts = text.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().lower()
                        value = parts[1].strip()
                        
                        if 'height' in key and not stats['height']:
                            stats['height'] = value if value != '--' else None
                        elif 'weight' in key and not stats['weight']:
                            stats['weight'] = value if value != '--' else None
                        elif 'reach' in key and not stats['reach']:
                            stats['reach'] = value if value != '--' else None
                        elif 'stance' in key and not stats['stance']:
                            stats['stance'] = value if value != '--' else None
                        elif 'dob' in key and not stats['dob']:
                            stats['dob'] = value if value != '--' else None
        
        # Parse career statistics from the table
        stats_table = soup.find('table', class_='b-fight-details__table b-fight-details__table_style_margin-top b-fight-details__table_type_event-details js-fight-table')
        
        if stats_table:
            rows = stats_table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    for i in range(len(cells) - 1):
                        cell_text = cells[i].get_text(strip=True).lower()
                        value_text = cells[i + 1].get_text(strip=True)
                        
                        try:
                            if 'slpm' in cell_text:
                                stats['slpm'] = float(value_text) if value_text != '--' else None
                            elif 'str. acc' in cell_text:
                                stats['str_acc'] = float(value_text.replace('%', '')) / 100 if '%' in value_text and value_text != '--' else None
                            elif 'sapm' in cell_text:
                                stats['sapm'] = float(value_text) if value_text != '--' else None
                            elif 'str. def' in cell_text:
                                stats['str_def'] = float(value_text.replace('%', '')) / 100 if '%' in value_text and value_text != '--' else None
                            elif 'td avg' in cell_text:
                                stats['td_avg'] = float(value_text) if value_text != '--' else None
                            elif 'td acc' in cell_text:
                                stats['td_acc'] = float(value_text.replace('%', '')) / 100 if '%' in value_text and value_text != '--' else None
                            elif 'td def' in cell_text:
                                stats['td_def'] = float(value_text.replace('%', '')) / 100 if '%' in value_text and value_text != '--' else None
                            elif 'sub. avg' in cell_text:
                                stats['sub_avg'] = float(value_text) if value_text != '--' else None
                        except (ValueError, AttributeError):
                            continue
        
        # Alternative parsing for stats if table method doesn't work
        if stats['slpm'] is None:  # Try multiple alternative methods
            
            # Method 1: Look for career statistics in specific classes
            career_stats = soup.find_all(['p', 'li', 'div'])
            for stat in career_stats:
                text = stat.get_text(strip=True)
                try:
                    # Check for colon-separated stats
                    if ':' in text:
                        # Split on colon and look for stat patterns
                        parts = text.split(':')
                        if len(parts) >= 2:
                            key = parts[0].strip().lower()
                            value = parts[1].strip().split()[0]  # Take first word after colon
                            
                            if 'slpm' in key and stats['slpm'] is None:
                                stats['slpm'] = float(value) if value != '--' else None
                            elif 'str. acc' in key and stats['str_acc'] is None:
                                clean_val = value.replace('%', '')
                                stats['str_acc'] = float(clean_val) / 100 if clean_val != '--' else None
                            elif 'sapm' in key and stats['sapm'] is None:
                                stats['sapm'] = float(value) if value != '--' else None
                            elif 'str. def' in key and stats['str_def'] is None:
                                clean_val = value.replace('%', '')
                                stats['str_def'] = float(clean_val) / 100 if clean_val != '--' else None
                            elif 'td avg' in key and stats['td_avg'] is None:
                                stats['td_avg'] = float(value) if value != '--' else None
                            elif 'td acc' in key and stats['td_acc'] is None:
                                clean_val = value.replace('%', '')
                                stats['td_acc'] = float(clean_val) / 100 if clean_val != '--' else None
                            elif 'td def' in key and stats['td_def'] is None:
                                clean_val = value.replace('%', '')
                                stats['td_def'] = float(clean_val) / 100 if clean_val != '--' else None
                            elif 'sub. avg' in key and stats['sub_avg'] is None:
                                stats['sub_avg'] = float(value) if value != '--' else None
                                
                    # Also try direct text matching
                    if 'SLpM:' in text and stats['slpm'] is None:
                        value = text.split('SLpM:')[1].strip().split()[0]
                        stats['slpm'] = float(value) if value != '--' else None
                    elif 'Str. Acc.:' in text and stats['str_acc'] is None:
                        value = text.split('Str. Acc.:')[1].strip().split()[0].replace('%', '')
                        stats['str_acc'] = float(value) / 100 if value != '--' else None
                    elif 'SApM:' in text and stats['sapm'] is None:
                        value = text.split('SApM:')[1].strip().split()[0]
                        stats['sapm'] = float(value) if value != '--' else None
                    elif 'Str. Def.:' in text and stats['str_def'] is None:
                        value = text.split('Str. Def.:')[1].strip().split()[0].replace('%', '')
                        stats['str_def'] = float(value) / 100 if value != '--' else None
                    elif 'TD Avg.:' in text and stats['td_avg'] is None:
                        value = text.split('TD Avg.:')[1].strip().split()[0]
                        stats['td_avg'] = float(value) if value != '--' else None
                    elif 'TD Acc.:' in text and stats['td_acc'] is None:
                        value = text.split('TD Acc.:')[1].strip().split()[0].replace('%', '')
                        stats['td_acc'] = float(value) / 100 if value != '--' else None
                    elif 'TD Def.:' in text and stats['td_def'] is None:
                        value = text.split('TD Def.:')[1].strip().split()[0].replace('%', '')
                        stats['td_def'] = float(value) / 100 if value != '--' else None
                    elif 'Sub. Avg.:' in text and stats['sub_avg'] is None:
                        value = text.split('Sub. Avg.:')[1].strip().split()[0]
                        stats['sub_avg'] = float(value) if value != '--' else None
                        
                except (ValueError, AttributeError, IndexError):
                    continue
        
        return stats
    
    def scrape_fighters_list_only(self):
        fighters = self.get_fighters_list()
        self.save_fighters_to_db(fighters)
        print(f"Saved {len(fighters)} fighters to database")
        
    def scrape_all_fighters(self, progress_callback=None, limit=None):
        # Get and save fighter list with progress tracking
        fighters = self.get_fighters_list(progress_callback)
        self.save_fighters_to_db(fighters)
        
        if progress_callback:
            progress_callback(f"Found {len(fighters)} fighters in database", 0, len(fighters))
        
        # Scrape individual fighter stats
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if limit:
            cursor.execute('SELECT id, full_name, url FROM fighters WHERE scraped = FALSE LIMIT ?', (limit,))
        else:
            cursor.execute('SELECT id, full_name, url FROM fighters WHERE scraped = FALSE')
        unscraped_fighters = cursor.fetchall()
        
        total_to_scrape = len(unscraped_fighters)
        scraped_count = 0
        errors = []
        
        if progress_callback:
            progress_callback(f"Starting to scrape {total_to_scrape} fighters...", scraped_count, total_to_scrape)
        
        for fighter_id, name, url in unscraped_fighters:
            try:
                if progress_callback:
                    progress_callback(f"Scraping: {name}", scraped_count, total_to_scrape)
                
                stats = self.get_fighter_stats(url)
                
                # Save stats to database
                cursor.execute('''
                    INSERT OR REPLACE INTO fighter_stats 
                    (fighter_id, height, weight, reach, stance, dob, wins, losses, draws,
                     slpm, str_acc, sapm, str_def, td_avg, td_acc, td_def, sub_avg)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    fighter_id,
                    stats.get('height'),
                    stats.get('weight'), 
                    stats.get('reach'),
                    stats.get('stance'),
                    stats.get('dob'),
                    stats.get('wins'),
                    stats.get('losses'),
                    stats.get('draws'),
                    stats.get('slpm'),
                    stats.get('str_acc'),
                    stats.get('sapm'),
                    stats.get('str_def'),
                    stats.get('td_avg'),
                    stats.get('td_acc'),
                    stats.get('td_def'),
                    stats.get('sub_avg')
                ))
                
                # Mark as scraped
                cursor.execute('UPDATE fighters SET scraped = TRUE WHERE id = ?', (fighter_id,))
                conn.commit()
                
                scraped_count += 1
                if progress_callback:
                    progress_callback(f"✓ Scraped: {name}", scraped_count, total_to_scrape)
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                error_msg = f"✗ Error scraping {name}: {str(e)}"
                errors.append(error_msg)
                if progress_callback:
                    progress_callback(error_msg, scraped_count, total_to_scrape)
                continue
        
        conn.close()
        
        if progress_callback:
            success_msg = f"Scraping completed! {scraped_count} fighters scraped successfully"
            if errors:
                success_msg += f", {len(errors)} errors"
            progress_callback(success_msg, scraped_count, total_to_scrape)

if __name__ == "__main__":
    scraper = UFCScraper()
    scraper.scrape_all_fighters()