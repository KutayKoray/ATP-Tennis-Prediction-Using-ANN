import pandas as pd
import re

# Read the file
with open("IBM's predictions.txt", 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Parse matches
matches = []
current_round = None

for line in lines:
    line = line.strip()
    
    # Detect round
    if '📊' in line and 'Round' in line or 'Quarter' in line or 'Semi' in line or 'Final' in line:
        # Extract round name
        current_round = line.split('📊')[1].strip() if '📊' in line else line
        continue
    
    # Skip headers and separator lines
    if not line or line.startswith('=') or line.startswith('-') or \
       line.startswith('Player 1') or line.startswith('Accuracy') or \
       line.startswith('✓') or line.startswith('🎾') or \
       line.startswith('Total') or line.startswith('Correct') or \
       line.startswith('Overall') or '✅' in line:
        continue
    
    # Parse match line
    # Format: Player1  Player2  Act  Pred  P1%  P2%  Conf  ✓  IBMp1%  IBMp2%
    parts = line.split()
    
    # Find indices of key markers
    try:
        # Find where P1/P2 (Act) starts - it should be after player names
        p_indices = [i for i, p in enumerate(parts) if p.startswith('P') and len(p) == 2]
        
        if len(p_indices) < 2:
            continue
            
        # Act is first P1/P2
        act_idx = p_indices[0]
        
        # Find IBM percentages (they start with %)
        ibm_indices = [i for i, p in enumerate(parts) if p.startswith('%')]
        
        if len(ibm_indices) < 1:
            continue
        
        # Extract player names (everything before Act)
        player_names = ' '.join(parts[:act_idx])
        
        # Split player names - find the boundary
        # Strategy: Look for capital letter after lowercase (new name starts)
        words = parts[:act_idx]
        
        # Find split point - when we see a pattern that suggests new name
        split_idx = None
        for i in range(1, len(words)):
            # Check if current word starts with capital and previous ends with lowercase
            if words[i][0].isupper():
                # This could be start of player 2
                # Check if it makes sense (both halves have reasonable length)
                p1_words = words[:i]
                p2_words = words[i:]
                if len(p1_words) >= 1 and len(p2_words) >= 1:
                    # Additional check: if we have 2 words each, likely correct
                    if len(p1_words) >= 2 and len(p2_words) >= 2:
                        split_idx = i
                        break
                    # Or if we have 3+ words total and reasonable split
                    elif len(words) >= 3:
                        split_idx = i
        
        # Fallback: split in middle
        if split_idx is None:
            split_idx = len(words) // 2
        
        player1 = ' '.join(words[:split_idx])
        player2 = ' '.join(words[split_idx:act_idx])
        
        # Extract Act (actual winner)
        actual_winner = parts[act_idx]
        
        # Extract IBM percentages
        ibm_p1_str = parts[ibm_indices[0]]  # First % is IBMp1%
        ibm_p1 = int(ibm_p1_str.replace('%', '')) if ibm_p1_str != '%' else None
        
        # Calculate P2%
        if ibm_p1 is not None:
            ibm_p2 = 100 - ibm_p1
        else:
            ibm_p2 = None
        
        # Determine IBM's prediction
        if ibm_p1 is not None and ibm_p2 is not None:
            ibm_pred = 'P1' if ibm_p1 > ibm_p2 else 'P2'
            ibm_conf = max(ibm_p1, ibm_p2)
        else:
            ibm_pred = None
            ibm_conf = None
        
        matches.append({
            'round': current_round,
            'player1': player1,
            'player2': player2,
            'actual_winner': actual_winner,
            'ibm_prediction': ibm_pred,
            'ibm_p1_probability': ibm_p1,
            'ibm_p2_probability': ibm_p2,
            'ibm_confidence': ibm_conf
        })
        
    except Exception as e:
        # Skip problematic lines
        continue

# Create DataFrame
df = pd.DataFrame(matches)

# Clean up and format
df_final = df[['player1', 'player2', 'actual_winner', 'ibm_prediction', 
               'ibm_p1_probability', 'ibm_p2_probability', 'ibm_confidence']].copy()

df_final.columns = ['Player_1', 'Player_2', 'Act', 'IBM_Pred', 'P1%', 'P2%', 'Conf']

# Save to CSV
df_final.to_csv("IBM's_predictions.csv", index=False)

print(f"✅ Created IBM's_predictions.csv with {len(df_final)} matches")
print(f"\nFirst 5 rows:")
print(df_final.head())
print(f"\nLast 5 rows:")
print(df_final.tail())
print(f"\nSummary:")
print(f"Total matches: {len(df_final)}")
print(f"Matches with predictions: {df_final['IBM_Pred'].notna().sum()}")
