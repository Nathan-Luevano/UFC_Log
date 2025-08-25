def parse_ply():
    player_stats = []
    with open('players.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.count('-') == 2:
                parts = line.strip().split('-')
                if len(parts) == 3 and all(part.isdigit() for part in parts):
                    player_stats.append({
                        'attack': int(parts[0]),
                        'defense': int(parts[1]), 
                        'speed': int(parts[2])
                    })
            else:
                continue
                
    return player_stats

stats = parse_ply()