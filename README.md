# UFC Predictor v1.0

High-performance UFC fight prediction system with real web scraping and optimized CLI interface.

## Quick Start

```bash
python3 ufc_predictor.py
```

## What's New in v1.0

- **Real Web Scraping**: Scrapes live data from ufcstats.com with 8 concurrent workers
- **Complete Fighter Database**: Full UFC roster with detailed stats  
- **Optimized Performance**: 600+ fighters in 10-15 minutes
- **Streamlined Interface**: 6 focused options instead of 8 redundant ones
- **Complete Fighter Stats**: All data in one view (record, physical, striking, grappling)
- **Smart Search**: Fuzzy matching with multiple fallback strategies

## Project Structure

```
UFC_Log/
├── ufc_predictor.py        # Main CLI (start here)
├── ufc_scraper.py          # High-performance web scraper  
├── database.py             # Fighter database management
├── simple_model.py         # Prediction algorithm
├── fight_predictor.py      # Command-line tool
├── data/
│   └── ufc_database.json   # Fighter data storage
└── TECHNICAL_DOCS.md       # Full technical documentation
```

## Features

### Interactive CLI
```
██╗   ██╗███████╗ ██████╗    ██████╗ ██████╗ ███████╗██████╗ 
██║   ██║██╔════╝██╔════╝    ██╔══██╗██╔══██╗██╔════╝██╔══██╗
██║   ██║█████╗  ██║         ██████╔╝██████╔╝█████╗  ██║  ██║
██║   ██║██╔══╝  ██║         ██╔═══╝ ██╔══██╗██╔══╝  ██║  ██║
╚██████╔╝██║     ╚██████╗    ██║     ██║  ██║███████╗██████╔╝
 ╚═════╝ ╚═╝      ╚═════╝    ╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ 

     Version      2.0.0
    Fighters      1,247
     Active       1,156  
Total Fights      15,432
     Updated      2023-08-30
       Model      Scoring Algorithm v2
      Status      Ready for Predictions
```

### Main Options
1. **Fighter Search & Stats** - Find fighters and see complete profiles
2. **Predict Fight** - Head-to-head predictions with confidence levels
3. **Browse All Fighters** - Sorted list with detailed stats on demand
4. **Update Database** - Quick (50 fighters) or Full (all UFC fighters) 
5. **Random Fight** - Generate random matchups with predictions
6. **Exit**

### Complete Fighter Stats
```
╔═══ Jon Jones ═══════════════════════════════════════════╗
║
║ RECORD & PHYSICAL
║   Record:       27-1-1
║   Age:          38 years old
║   Height:       76"
║   Weight:       205 lbs
║   Reach:        84"
║   Stance:       Orthodox
║
║ STRIKING STATS
║   Strikes/Min:  4.29 landed, 2.05 absorbed
║   Accuracy:     58.0%
║   Defense:      62.0%
║
║ GRAPPLING STATS  
║   Takedowns:    2.07 per fight
║   TD Accuracy:  43.0%
║   TD Defense:   95.0%
║   Submissions:  0.40 per fight
╚═══════════════════════════════════════════════════════════╝
```

## High-Performance Web Scraping

### Features
- **Concurrent Processing**: 8 simultaneous connections
- **Smart Rate Limiting**: 0.5-1.5 second random delays
- **Automatic Retry**: Exponential backoff on failures
- **Progress Tracking**: Real-time scraping status
- **Data Validation**: Robust parsing with fallbacks

### Performance
- **Quick Update**: 50 fighters in ~2 minutes  
- **Full Update**: 600+ fighters in 10-15 minutes
- **Success Rate**: >95% with built-in error handling
- **Memory Usage**: ~750KB for full UFC roster

## Installation & Setup

### Requirements
- Python 3.7+
- No external dependencies (uses only standard library)
- Internet connection for scraping
- ~10MB disk space

### First Time Setup
1. Clone repository
2. Run `python3 ufc_predictor.py`
3. Choose option 4 "Update Database" 
4. Select "Quick update" for testing or "Full update" for complete data
5. Start making predictions!

## Files Overview

- **ufc_predictor.py** - Main interface with ASCII art and system info
- **ufc_scraper.py** - High-performance concurrent web scraper
- **database.py** - Fighter database with fuzzy search
- **simple_model.py** - Lightweight prediction algorithm
- **fight_predictor.py** - Command-line tool for scripting

## Technical Documentation

See `TECHNICAL_DOCS.md` for complete architecture, API reference, and contributing guidelines.