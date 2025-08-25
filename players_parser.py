import re
from datetime import datetime, date
from typing import Dict, Any, Tuple


def _to_float(s: str, percent: bool = False) -> float:
    if s is None:
        return 0.0
    s = s.strip()
    if s.endswith('%'):
        try:
            return float(s.rstrip('%')) / 100.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _height_to_inches(h: str) -> int:
    if not h:
        return 0
    h = h.replace('"', '"').replace('”', '"')
    m = re.search(r"(\d+)\s*'\s*(\d+)", h)
    if m:
        feet = int(m.group(1))
        inches = int(m.group(2))
        return feet * 12 + inches
    m2 = re.search(r"(\d+)\s*ft(?:eet)?\s*(\d+)\s*in", h)
    if m2:
        return int(m2.group(1)) * 12 + int(m2.group(2))
    m3 = re.search(r"(\d+)", h)
    return int(m3.group(1)) if m3 else 0


def _parse_record(rec_line: str) -> Tuple[int, int]:
    m = re.search(r"(\d+)-(\d+)-?(\d+)?", rec_line)
    if m:
        wins = int(m.group(1))
        losses = int(m.group(2))
        return wins, losses
    return 0, 0


def _compute_age(dob_line: str) -> int:
    if not dob_line:
        return 0
    dob_line = dob_line.strip()
    try:
        dob = datetime.strptime(dob_line, '%b %d, %Y').date()
    except ValueError:
        try:
            dob = datetime.strptime(dob_line, '%B %d, %Y').date()
        except ValueError:
            return 0
    today = date.today()
    years = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    return years


def parse_players_file(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()

    raw = raw.strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw = '\n'.join(raw.split('\n')[1:-1])

    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    players = {}
    current = None
    buffer = []

    def flush_player(name, buf):
        if not name:
            return
        content = '\n'.join(buf)
        stats = {
            'wins_total': 0,
            'losses_total': 0,
            'age': 0,
            'height': 0,
            'weight': 0,
            'reach': 0,
            'stance': 'Unknown',
            'SLpM_total': 0.0,
            'SApM_total': 0.0,
            'sig_str_acc_total': 0.0,
            'td_acc_total': 0.0,
            'str_def_total': 0.0,
            'td_def_total': 0.0,
            'sub_avg': 0.0,
            'td_avg': 0.0
        }

        # Record
        rec_lines = [l for l in buf if 'Record:' in l]
        if rec_lines:
            wins, losses = _parse_record(rec_lines[0])
            stats['wins_total'] = wins
            stats['losses_total'] = losses

        # Height
        m = re.search(r'Height:\s*(.+)', content)
        if m:
            stats['height'] = _height_to_inches(m.group(1))

        # Weight
        m = re.search(r'Weight:\s*([0-9]+)', content)
        if m:
            stats['weight'] = int(m.group(1))

        # Reach
        m = re.search(r'Reach:\s*([0-9]+)', content)
        if m:
            stats['reach'] = int(m.group(1))

        # Stance
        m = re.search(r'STANCE:\s*(\w+)', content, re.IGNORECASE)
        if m:
            stats['stance'] = m.group(1).capitalize()

        # DOB -> age
        m = re.search(r'DOB:\s*(.+)', content)
        if m:
            stats['age'] = _compute_age(m.group(1).strip())

        # SLpM
        m = re.search(r'SLpM:\s*([0-9\.]+)', content)
        if m:
            stats['SLpM_total'] = _to_float(m.group(1))

        # SApM
        m = re.search(r'SApM:\s*([0-9\.]+)', content)
        if m:
            stats['SApM_total'] = _to_float(m.group(1))

        # Str. Acc.
        m = re.search(r'Str\. Acc\.:\s*([0-9\.]+%?)', content)
        if m:
            stats['sig_str_acc_total'] = _to_float(m.group(1), percent=True)

        # Str. Def
        m = re.search(r'Str\. Def:\s*([0-9\.]+%?)', content)
        if m:
            stats['str_def_total'] = _to_float(m.group(1), percent=True)

        # TD Avg
        m = re.search(r'TD Avg\.:\s*([0-9\.]+)', content)
        if m:
            stats['td_avg'] = _to_float(m.group(1))

        # TD Acc
        m = re.search(r'TD Acc\.:\s*([0-9\.]+%?)', content)
        if m:
            stats['td_acc_total'] = _to_float(m.group(1), percent=True)

        # TD Def
        m = re.search(r'TD Def\.:\s*([0-9\.]+%?)', content)
        if m:
            stats['td_def_total'] = _to_float(m.group(1), percent=True)

        # Sub Avg
        m = re.search(r'Sub\. Avg\.:\s*([0-9\.]+)', content)
        if m:
            stats['sub_avg'] = _to_float(m.group(1))

        players[name] = stats

    for line in lines:
        if 'Record:' in line:
            name_part = line.split('Record:')[0].strip()
            if current:
                flush_player(current, buffer)
            current = name_part
            buffer = [line]
        else:
            if current is None:
                continue
            buffer.append(line)

    if current:
        flush_player(current, buffer)

    return players


def get_player_by_name(name: str, players: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    for n, stats in players.items():
        if n.lower() == name.lower():
            return stats
    for n, stats in players.items():
        if name.lower() in n.lower():
            return stats
    return {}


if __name__ == '__main__':
    import os
    path = os.path.join(os.path.dirname(__file__), 'players.txt')
    parsed = parse_players_file(path)
    for n, s in parsed.items():
        print(f"Player: {n}")
        print(s)

