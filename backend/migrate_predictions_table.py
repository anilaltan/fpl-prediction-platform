"""
Migration script to update Prediction table schema for Batch Prediction system.
Adds new columns and renames existing ones.
"""
from app.database import engine
from sqlalchemy import text

def migrate_predictions_table():
    """Migrate Prediction table to new schema"""
    with engine.connect() as conn:
        try:
            # Add new columns
            conn.execute(text("""
                ALTER TABLE predictions 
                ADD COLUMN IF NOT EXISTS fpl_id INTEGER;
            """))
            
            conn.execute(text("""
                ALTER TABLE predictions 
                ADD COLUMN IF NOT EXISTS season VARCHAR DEFAULT '2025-26';
            """))
            
            conn.execute(text("""
                ALTER TABLE predictions 
                ADD COLUMN IF NOT EXISTS xg FLOAT DEFAULT 0.0;
            """))
            
            conn.execute(text("""
                ALTER TABLE predictions 
                ADD COLUMN IF NOT EXISTS xa FLOAT DEFAULT 0.0;
            """))
            
            conn.execute(text("""
                ALTER TABLE predictions 
                ADD COLUMN IF NOT EXISTS xmins FLOAT DEFAULT 0.0;
            """))
            
            conn.execute(text("""
                ALTER TABLE predictions 
                ADD COLUMN IF NOT EXISTS xcs FLOAT DEFAULT 0.0;
            """))
            
            conn.execute(text("""
                ALTER TABLE predictions 
                ADD COLUMN IF NOT EXISTS defcon_score FLOAT DEFAULT 0.0;
            """))
            
            conn.execute(text("""
                ALTER TABLE predictions 
                ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP;
            """))
            
            # Rename columns if they exist and new names don't exist
            # Check if xp column exists, if not rename predicted_points
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='predictions' AND column_name='xp'
            """))
            if result.fetchone() is None:
                # Check if predicted_points exists
                result2 = conn.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='predictions' AND column_name='predicted_points'
                """))
                if result2.fetchone():
                    conn.execute(text("""
                        ALTER TABLE predictions 
                        RENAME COLUMN predicted_points TO xp;
                    """))
            
            # Rename created_at to calculated_at if needed
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name='predictions' AND column_name='calculated_at'
            """))
            if result.fetchone() is None:
                result2 = conn.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='predictions' AND column_name='created_at'
                """))
                if result2.fetchone():
                    conn.execute(text("""
                        ALTER TABLE predictions 
                        RENAME COLUMN created_at TO calculated_at;
                    """))
            
            # Populate fpl_id from player_id if fpl_id is NULL
            # Note: players.id is the FPL player ID (primary key)
            conn.execute(text("""
                UPDATE predictions p
                SET fpl_id = (
                    SELECT pl.id 
                    FROM players pl 
                    WHERE pl.id = p.player_id
                )
                WHERE p.fpl_id IS NULL AND p.player_id IS NOT NULL;
            """))
            
            conn.commit()
            print("✅ Migration completed successfully")
            
        except Exception as e:
            conn.rollback()
            print(f"❌ Migration failed: {str(e)}")
            raise

if __name__ == "__main__":
    migrate_predictions_table()
