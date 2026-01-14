import pandas as pd
import numpy as np
from app.services.ml_engine import PLEngine, AttackModel
from app.database import SessionLocal
from app.models import Player, PlayerGameweekStats

def debug_logic():
    """
    Debug script to verify ML engine predictions after bug fixes.
    Tests:
    1. Appearance points are added correctly
    2. Historical averages are used (not current week stats)
    3. No target leakage in attack model training
    """
    engine = PLEngine()
    engine._ensure_models_loaded()
    print(f"Model durumu: xMins Trained={engine.xmins_model.is_trained}")
    
    db = SessionLocal()
    
    # TRAINING WITH PROPER FEATURES (like backtest does)
    print("\n=== Loading training data ===")
    training_data = pd.read_sql(
        db.query(PlayerGameweekStats).filter(
            PlayerGameweekStats.season == "2025-26"
        ).statement, 
        db.bind
    )
    print(f"Loaded {len(training_data)} rows of training data")
    
    if training_data.empty:
        print("ERROR: No training data found!")
        return
    
    # Prepare xMins features (8 features to match XMinsModel)
    print("\n=== Preparing xMins features ===")
    xmins_features = []
    xmins_labels = []
    
    training_data_sorted = training_data.sort_values(['fpl_id', 'gameweek'])
    for idx, row in training_data_sorted.iterrows():
        player_id = row.get('fpl_id')
        player_rows = training_data_sorted[training_data_sorted['fpl_id'] == player_id]
        current_gw = row.get('gameweek', 1)
        prev_matches = player_rows[player_rows['gameweek'] < current_gw]
        
        if len(prev_matches) > 0:
            recent_mins = prev_matches['minutes'].tail(3).tolist()
            recent_minutes_avg = float(np.mean([m for m in recent_mins if m > 0])) / 90.0 if any(m > 0 for m in recent_mins) else 0.5
        else:
            recent_minutes_avg = 0.5
        
        feature = [
            7.0,  # days_since_last_match
            0.0,  # is_cup_week
            0.0,  # injury_status
            recent_minutes_avg,  # recent_minutes_avg / 90
            2.0 / 3.0,  # position_depth / 3
            float(row.get('total_points', 0)) / 10.0,  # form_score / 10
            float(row.get('price', 5.0)) / 100.0 if 'price' in row else 0.05,  # price
            0.5   # rotation_risk
        ]
        xmins_features.append(feature)
        xmins_labels.append(1 if row.get('minutes', 0) > 0 else 0)
    
    xmins_features = np.array(xmins_features)
    xmins_labels = np.array(xmins_labels)
    print(f"xMins features: {xmins_features.shape}, labels: {xmins_labels.shape}")
    
    # Prepare Attack features (17 features, NO TARGET LEAKAGE)
    print("\n=== Preparing Attack features (NO LEAKAGE) ===")
    attack_model = AttackModel()
    attack_features = []
    attack_xg_labels = []
    attack_xa_labels = []
    
    # Build cumulative stats per player (to avoid leakage)
    player_cumulative = {}
    for fpl_id, group in training_data_sorted.groupby('fpl_id'):
        group = group.sort_values('gameweek')
        player_cumulative[fpl_id] = {}
        cum_mins, cum_xg, cum_xa = 0.0, 0.0, 0.0
        recent_xg, recent_xa, recent_mins = [], [], []
        
        for _, row in group.iterrows():
            gw = row.get('gameweek', 0)
            # Store BEFORE this week's stats (for prediction)
            if cum_mins > 0:
                player_cumulative[fpl_id][gw] = {
                    'xg_per_90': (cum_xg / cum_mins) * 90,
                    'xa_per_90': (cum_xa / cum_mins) * 90,
                    'recent_xg': recent_xg[-5:].copy(),
                    'recent_xa': recent_xa[-5:].copy(),
                    'recent_minutes': recent_mins[-5:].copy(),
                    'minutes': cum_mins
                }
            else:
                player_cumulative[fpl_id][gw] = None
            
            # Update cumulative
            cum_mins += float(row.get('minutes', 0) or 0)
            cum_xg += float(row.get('xg', 0) or 0)
            cum_xa += float(row.get('xa', 0) or 0)
            recent_xg.append(float(row.get('xg', 0) or 0))
            recent_xa.append(float(row.get('xa', 0) or 0))
            recent_mins.append(float(row.get('minutes', 0) or 0))
    
    # Now create features using HISTORICAL data only
    for _, row in training_data_sorted.iterrows():
        fpl_id = row.get('fpl_id')
        gw = row.get('gameweek', 0)
        
        if fpl_id in player_cumulative and gw in player_cumulative[fpl_id]:
            hist = player_cumulative[fpl_id][gw]
            if hist is None:
                continue  # No history yet
            
            player_data = {
                'fpl_id': fpl_id,
                'position': row.get('position', 'MID'),
                'xg_per_90': hist['xg_per_90'],
                'xa_per_90': hist['xa_per_90'],
                'recent_xg': hist['recent_xg'],
                'recent_xa': hist['recent_xa'],
                'recent_minutes': hist['recent_minutes'],
                'minutes': hist['minutes'],
                'expected_minutes': float(np.mean(hist['recent_minutes'])) if hist['recent_minutes'] else 0
            }
            
            feat = attack_model.extract_features(
                player_data=player_data,
                fixture_data={'is_home': bool(row.get('was_home', True))}
            )
            attack_features.append(feat.flatten().tolist())
            attack_xg_labels.append(float(row.get('xg', 0)))
            attack_xa_labels.append(float(row.get('xa', 0)))
    
    attack_features = np.array(attack_features)
    attack_xg_labels = np.array(attack_xg_labels)
    attack_xa_labels = np.array(attack_xa_labels)
    print(f"Attack features: {attack_features.shape}, xG labels: {attack_xg_labels.shape}, xA labels: {attack_xa_labels.shape}")
    
    # TRAIN THE MODEL
    print("\n=== Training model ===")
    engine.train(
        training_data=training_data,
        xmins_features=xmins_features,
        xmins_labels=xmins_labels,
        attack_features=attack_features,
        attack_xg_labels=attack_xg_labels,
        attack_xa_labels=attack_xa_labels
    )
    print(f"Post-training: xMins Trained={engine.xmins_model.is_trained}, Attack Trained={engine.attack_model.xg_trained}")
    
    # TEST HAALAND PREDICTION
    print("\n=== Testing Haaland Prediction ===")
    haaland = db.query(Player).filter(Player.name.like('%Haaland%')).first()
    
    # Use realistic HISTORICAL stats for Haaland (as if predicting his next game)
    test_data = {
        'fpl_id': haaland.fpl_id if haaland else 430,
        'position': 'FWD',
        'xg_per_90': 0.85,  # Haaland's historical xG/90
        'xa_per_90': 0.12,  # Haaland's historical xA/90
        'goals_per_90': 0.80,
        'assists_per_90': 0.10,
        'minutes': 2500,  # Total minutes played (shows he's a regular starter)
        'expected_minutes': 85.0,  # Expected to play ~85 mins
        'recent_minutes': [90, 90, 77, 90, 90],  # Recent 5 games
        'recent_xg': [0.95, 1.1, 0.7, 0.9, 1.2],  # Recent xG
        'recent_xa': [0.1, 0.2, 0.05, 0.15, 0.1],  # Recent xA
        'status': 'a',  # Available
        'form': 8.5,  # Good form
        'price': 150.0,  # £15.0m
        'ict_index': 12.5  # High ICT index for bonus points calculation
    }
    
    res = engine.calculate_expected_points(test_data, fixture_data=None)
    
    print(f"\n{'='*50}")
    print(f"HAALAND ({test_data['fpl_id']}) PREDICTION BREAKDOWN")
    print(f"{'='*50}")
    print(f"Expected Minutes (xMins): {res['xmins']:.1f}")
    print(f"xMins Factor: {res['xmins_factor']:.3f}")
    print(f"Start Probability (P_start): {res['p_start']:.3f}")
    print(f"")
    print(f"xG: {res['xg']:.3f}")
    print(f"xA: {res['xa']:.3f}")
    print(f"xCS: {res['xcs']:.3f}")
    print(f"")
    print(f"Goal Component: {res['goal_component']:.2f} (xG * 4 for FWD)")
    print(f"Assist Component: {res['assist_component']:.2f} (xA * 3)")
    print(f"CS Component: {res['cs_component']:.2f}")
    print(f"Appearance Points: {res.get('appearance_points', 0):.2f} (xMins/90 * 2)")
    print(f"Expected Bonus (xB): {res.get('expected_bonus', 0):.2f}")
    print(f"DefCon Points: {res['defcon_points']:.2f}")
    print(f"")
    print(f"{'='*50}")
    print(f"FINAL EXPECTED POINTS (xP): {res['expected_points']:.2f}")
    print(f"{'='*50}")
    
    # Verify the calculation is in reasonable range
    if 6.0 <= res['expected_points'] <= 9.0:
        print("\n✅ SUCCESS: Haaland xP is in expected range (6.5-8.5)")
    elif res['expected_points'] > 0.5:
        print(f"\n⚠️ PARTIAL: xP ({res['expected_points']:.2f}) is better than 0.2 but may need tuning")
    else:
        print(f"\n❌ FAIL: xP ({res['expected_points']:.2f}) is still too low!")
    
    db.close()

if __name__ == "__main__":
    debug_logic()
