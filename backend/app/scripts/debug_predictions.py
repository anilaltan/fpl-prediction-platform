import pandas as pd
from app.services.ml_engine import PLEngine
from app.database import SessionLocal
from app.models import Player
import gc

def audit_predictions():
    # Modelleri yüklemek için engine başlat
    engine = PLEngine()
    db = SessionLocal()
    
    # Test için en iyi oyuncuları al
    top_players = db.query(Player).order_by(Player.total_points.desc()).limit(10).all()
    
    print(f"\n{'Player':<20} | {'xMins':<6} | {'xG':<5} | {'xA':<5} | {'xCS':<5} | {'FINAL xP'}")
    print("-" * 75)

    for p in top_players:
        try:
            # Player objesini Dict'e dönüştür
            player_data = {
                'fpl_id': p.fpl_id,
                'name': p.name,
                'position': p.position,
                'price': float(p.price),
                'team': p.team,
                'total_points': p.total_points,
                'minutes': 0,  # Default values for missing fields
                'xg': 0.0,
                'xa': 0.0,
                'goals': 0,
                'assists': 0,
                'blocks': 0,
                'tackles': 0,
                'interceptions': 0,
                'passes': 0,
            }
            
            # calculate_expected_points metodu Dict bekliyor
            res = engine.calculate_expected_points(player_data, None)
            
            # Dict dönüyor, parçala
            xp = res.get('expected_points', 0)
            mins = res.get('xmins', 0)
            xg = res.get('xg', 0)
            xa = res.get('xa', 0)
            xcs = res.get('xcs', 0)

            print(f"{p.name:<20} | {mins:<6.1f} | {xg:<5.2f} | {xa:<5.2f} | {xcs:<5.2f} | {xp:<8.2f}")
                
        except Exception as e:
            print(f"{p.name:<20} | Hata: {str(e)}")
            import traceback
            traceback.print_exc()

    db.close()

if __name__ == "__main__":
    audit_predictions()