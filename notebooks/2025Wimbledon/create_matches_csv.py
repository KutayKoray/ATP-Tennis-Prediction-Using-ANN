import json
import csv
import os
from pathlib import Path

def extract_matches_from_json(json_file_path):
    """
    Extract match information from a Day JSON file.
    
    Returns a list of dictionaries with keys: round, p1_name, p2_name, winner
    """
    matches = []
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Iterate through all matches in the day
    for match in data.get('matches', []):
        # Extract round code
        round_code = match.get('roundCode', '')
        
        # Extract winner (1 or 2)
        winner = match.get('winner', '')
        
        # Extract player 1 name from team1
        team1 = match.get('team1', {})
        first_name_1 = team1.get('firstNameA', '')
        last_name_1 = team1.get('lastNameA', '')
        p1_name = f"{first_name_1} {last_name_1}".strip()
        
        # Extract player 2 name from team2
        team2 = match.get('team2', {})
        first_name_2 = team2.get('firstNameA', '')
        last_name_2 = team2.get('lastNameA', '')
        p2_name = f"{first_name_2} {last_name_2}".strip()
        
        # Add match to list
        matches.append({
            'round': round_code,
            'p1_name': p1_name,
            'p2_name': p2_name,
            'winner': winner
        })
    
    return matches

def process_all_days(wimbledon_folder):
    """
    Process all Day1-14.json files and combine matches into a single list.
    """
    all_matches = []
    
    # Process Day1 through Day14
    for day_num in range(1, 15):
        json_file = os.path.join(wimbledon_folder, f'Day{day_num}.json')
        
        if os.path.exists(json_file):
            print(f"Processing {json_file}...")
            matches = extract_matches_from_json(json_file)
            all_matches.extend(matches)
            print(f"  Found {len(matches)} matches")
        else:
            print(f"Warning: {json_file} not found, skipping...")
    
    return all_matches

def save_to_csv(matches, output_file):
    """
    Save matches to a CSV file.
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['round', 'p1_name', 'p2_name', 'winner']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(matches)
    
    print(f"\nCSV file created: {output_file}")
    print(f"Total matches: {len(matches)}")

def main():
    # Set the path to the 2025Wimbledon folder
    wimbledon_folder = Path(__file__).parent
    
    # Process all day files
    all_matches = process_all_days(wimbledon_folder)
    
    # Save to CSV
    output_file = wimbledon_folder / '2025_wimbledon_matches.csv'
    save_to_csv(all_matches, output_file)

if __name__ == '__main__':
    main()
