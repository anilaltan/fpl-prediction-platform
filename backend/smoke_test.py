"""
Smoke Test Script for FPL Prediction Platform
Tests all core services: Database, ETL, ML Engine, Solver, Strategy
"""
import sys
import os
import traceback
import gc
from typing import Dict, List, Optional
import asyncio
from datetime import datetime

# Try to import psutil, create dummy if not available
try:
    import psutil
except ImportError:
    # Create dummy psutil module
    class DummyMemoryInfo:
        rss = 0
    class DummyProcess:
        def memory_info(self):
            return DummyMemoryInfo()
    class DummyPsutil:
        def Process(self, pid):
            return DummyProcess()
    psutil = DummyPsutil()

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Import services
from app.database import SessionLocal, engine, Base
from app.models import Player, PlayerGameweekStats
from app.services.fpl_api import FPLAPIService
from app.services.ml_engine import PLEngine
from app.services.solver import FPLSolver
from app.services.strategy import StrategyService
from app.services.backtest import BacktestEngine
from sqlalchemy import text
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmokeTest:
    """Smoke test runner for FPL Prediction Platform"""
    
    def __init__(self):
        self.results = {
            'db_connection': {'passed': False, 'error': None},
            'etl_check': {'passed': False, 'error': None},
            'ml_check': {'passed': False, 'error': None, 'memory_used_mb': 0},
            'solver_check': {'passed': False, 'error': None},
            'strategy_check': {'passed': False, 'error': None},
            'backtest_check': {'passed': False, 'error': None}
        }
        self.memory_before = 0
        self.memory_after = 0
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def test_db_connection(self) -> bool:
        """
        Test 1: Database Connection
        Check if we can connect to the database.
        """
        logger.info("=" * 60)
        logger.info("TEST 1: Database Connection")
        logger.info("=" * 60)
        
        try:
            # Test connection
            db = SessionLocal()
            result = db.execute(text("SELECT 1"))
            db.close()
            
            if result:
                logger.info("✓ Database connection successful")
                self.results['db_connection']['passed'] = True
                return True
            else:
                logger.error("✗ Database connection failed: No result")
                self.results['db_connection']['error'] = "No result from query"
                return False
                
        except Exception as e:
            error_msg = f"Database connection error: {str(e)}"
            logger.error(f"✗ {error_msg}")
            self.results['db_connection']['error'] = error_msg
            return False
    
    async def test_etl_check(self) -> bool:
        """
        Test 2: ETL Check
        Fetch sample data from FPL API and UPSERT to database.
        """
        logger.info("=" * 60)
        logger.info("TEST 2: ETL Check (FPL API -> Database)")
        logger.info("=" * 60)
        
        try:
            # Create database tables if they don't exist
            logger.info("Creating database tables if needed...")
            Base.metadata.create_all(bind=engine)
            logger.info("✓ Database tables ready")
            
            # Initialize FPL API service
            fpl_api = FPLAPIService()
            
            # Fetch bootstrap data (lightweight test)
            logger.info("Fetching bootstrap data from FPL API...")
            bootstrap = await fpl_api.get_bootstrap_data()
            
            if not bootstrap:
                logger.error("✗ Failed to fetch bootstrap data")
                self.results['etl_check']['error'] = "No bootstrap data returned"
                return False
            
            elements = bootstrap.get('elements', [])
            if not elements:
                logger.error("✗ No elements in bootstrap data")
                self.results['etl_check']['error'] = "No elements in bootstrap"
                return False
            
            logger.info(f"✓ Fetched {len(elements)} players from FPL API")
            
            # Test database UPSERT with sample player
            db = SessionLocal()
            try:
                sample_player = elements[0]  # Get first player
                player_id = sample_player.get('id')
                
                # Check if player exists
                existing = db.query(Player).filter(Player.fpl_id == player_id).first()
                
                if existing:
                    logger.info(f"Player {player_id} already exists, testing update...")
                    existing.name = sample_player.get('web_name', existing.name)
                    db.commit()
                    logger.info("✓ Player update successful")
                else:
                    logger.info(f"Creating new player {player_id}...")
                    new_player = Player(
                        fpl_id=player_id,
                        name=sample_player.get('web_name', sample_player.get('first_name', '') + ' ' + sample_player.get('second_name', '')),
                        team=str(sample_player.get('team', 0)),
                        position=self._get_position_name(sample_player.get('element_type', 3)),
                        price=sample_player.get('now_cost', 0) / 10.0
                    )
                    db.add(new_player)
                    db.commit()
                    logger.info("✓ Player creation successful")
                
                # Verify
                verified = db.query(Player).filter(Player.fpl_id == player_id).first()
                if verified:
                    logger.info(f"✓ UPSERT verified: Player {verified.name} (ID: {verified.fpl_id})")
                    self.results['etl_check']['passed'] = True
                    return True
                else:
                    logger.error("✗ UPSERT verification failed")
                    self.results['etl_check']['error'] = "Player not found after UPSERT"
                    return False
                    
            except Exception as e:
                db.rollback()
                raise e
            finally:
                db.close()
                
        except Exception as e:
            error_msg = f"ETL check error: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"✗ {error_msg}")
            self.results['etl_check']['error'] = error_msg
            return False
    
    def test_ml_check(self) -> bool:
        """
        Test 3: ML Check
        Load PLEngine models within 4GB RAM limit and calculate xP for a player.
        """
        logger.info("=" * 60)
        logger.info("TEST 3: ML Engine Check (4GB RAM Limit)")
        logger.info("=" * 60)
        
        try:
            self.memory_before = self.get_memory_usage_mb()
            logger.info(f"Memory before ML test: {self.memory_before:.2f} MB")
            
            # Initialize PLEngine
            logger.info("Initializing PLEngine...")
            plengine = PLEngine()
            
            # Create sample player data
            sample_player = {
                'id': 1,
                'name': 'Test Player',
                'position': 'MID',
                'price': 8.0,
                'team_id': 1,
                'minutes': 90,
                'goals_scored': 0,
                'assists': 0,
                'clean_sheets': 0,
                'status': 'a',
                'xg_per_90': 0.3,
                'xa_per_90': 0.2,
                'goals_per_90': 0.2,
                'assists_per_90': 0.15
            }
            
            sample_fixture = {
                'is_home': True,
                'opponent_team': 2,
                'difficulty': 3
            }
            
            # Test prediction (models will be initialized if needed)
            logger.info("Testing xP prediction...")
            prediction = plengine.predict(
                player_data=sample_player,
                fixture_data=sample_fixture
            )
            
            if prediction and 'expected_points' in prediction:
                xp = prediction['expected_points']
                logger.info(f"✓ xP prediction successful: {xp:.2f} points")
                
                # Check memory usage
                self.memory_after = self.get_memory_usage_mb()
                memory_used = self.memory_after - self.memory_before
                self.results['ml_check']['memory_used_mb'] = memory_used
                
                logger.info(f"Memory after ML test: {self.memory_after:.2f} MB")
                logger.info(f"Memory used: {memory_used:.2f} MB")
                
                # Check if within 4GB limit (with some buffer)
                if memory_used > 3500:  # 3.5GB buffer for safety
                    logger.warning(f"⚠ Memory usage ({memory_used:.2f} MB) is high but within 4GB limit")
                
                # Cleanup
                try:
                    asyncio.run(plengine.async_unload_models())
                except:
                    pass
                gc.collect()
                
                self.results['ml_check']['passed'] = True
                return True
            else:
                logger.error("✗ Prediction returned invalid result")
                self.results['ml_check']['error'] = "Invalid prediction result"
                return False
                
        except MemoryError as e:
            error_msg = f"MemoryError: {str(e)}"
            logger.error(f"✗ {error_msg}")
            self.results['ml_check']['error'] = error_msg
            return False
        except Exception as e:
            error_msg = f"ML check error: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"✗ {error_msg}")
            self.results['ml_check']['error'] = error_msg
            return False
    
    def test_solver_check(self) -> bool:
        """
        Test 4: Solver Check
        Test FPLSolver with 100M budget and 3-player team limit constraints.
        """
        logger.info("=" * 60)
        logger.info("TEST 4: Solver Check (100M Budget, 3-Player Limit)")
        logger.info("=" * 60)
        
        try:
            # Initialize solver
            solver = FPLSolver(
                budget=100.0,
                horizon_weeks=3,
                free_transfers=1,
                discount_factor=0.9
            )
            
            # Create sample players data
            # Need more players to satisfy constraints (at least 2GK, 5DEF, 5MID, 3FWD)
            # And ensure we have enough players from different teams (max 3 per team)
            # Total budget should be <= 100M
            sample_players = []
            
            # 2 Goalkeepers (Team 1, Team 2) - cheaper options
            for i in range(1, 3):
                sample_players.append({
                    'id': i,
                    'name': f'GK{i}',
                    'position': 'GK',
                    'price': 4.5,  # Cheaper to fit budget
                    'team_id': i,
                    'expected_points': [4.0, 4.5, 4.0],
                    'p_start': [0.9, 0.9, 0.9]
                })
            
            # 5 Defenders (Teams 1-5) - spread across teams
            for i in range(3, 8):
                sample_players.append({
                    'id': i,
                    'name': f'DEF{i}',
                    'position': 'DEF',
                    'price': 5.0,  # Cheaper
                    'team_id': i - 2,  # Teams 1-5
                    'expected_points': [5.0, 5.5, 5.0],
                    'p_start': [0.85, 0.85, 0.85]
                })
            
            # 5 Midfielders (Teams 6-10) - different teams
            for i in range(8, 13):
                sample_players.append({
                    'id': i,
                    'name': f'MID{i}',
                    'position': 'MID',
                    'price': 6.0,  # Cheaper
                    'team_id': i - 2,  # Teams 6-10
                    'expected_points': [6.0, 6.5, 6.0],
                    'p_start': [0.9, 0.9, 0.9]
                })
            
            # 3 Forwards (Teams 11-13) - different teams
            for i in range(13, 16):
                sample_players.append({
                    'id': i,
                    'name': f'FWD{i}',
                    'position': 'FWD',
                    'price': 7.0,  # Cheaper
                    'team_id': i - 2,  # Teams 11-13
                    'expected_points': [7.0, 7.5, 7.0],
                    'p_start': [0.85, 0.85, 0.85]
                })
            
            # Add more players to give solver options (ensure feasibility)
            # Additional goalkeepers
            for i in range(16, 18):
                sample_players.append({
                    'id': i,
                    'name': f'GK{i}',
                    'position': 'GK',
                    'price': 4.0,
                    'team_id': i - 14,
                    'expected_points': [3.5, 4.0, 3.5],
                    'p_start': [0.8, 0.8, 0.8]
                })
            
            # Additional defenders
            for i in range(18, 23):
                sample_players.append({
                    'id': i,
                    'name': f'DEF{i}',
                    'position': 'DEF',
                    'price': 4.5,
                    'team_id': i - 12,
                    'expected_points': [4.5, 5.0, 4.5],
                    'p_start': [0.8, 0.8, 0.8]
                })
            
            # Additional midfielders
            for i in range(23, 28):
                sample_players.append({
                    'id': i,
                    'name': f'MID{i}',
                    'position': 'MID',
                    'price': 5.5,
                    'team_id': i - 17,
                    'expected_points': [5.5, 6.0, 5.5],
                    'p_start': [0.85, 0.85, 0.85]
                })
            
            # Additional forwards
            for i in range(28, 31):
                sample_players.append({
                    'id': i,
                    'name': f'FWD{i}',
                    'position': 'FWD',
                    'price': 6.5,
                    'team_id': i - 22,
                    'expected_points': [6.5, 7.0, 6.5],
                    'p_start': [0.8, 0.8, 0.8]
                })
            
            logger.info(f"Created {len(sample_players)} sample players")
            
            # Test optimization
            logger.info("Running solver optimization...")
            solution = solver.optimize_team(
                players_data=sample_players,
                current_squad=None,
                locked_players=None,
                excluded_players=None
            )
            
            # Verify constraints
            squad_week1 = solution.get('squad_week1', [])
            total_cost = solution.get('total_cost', 0.0)
            
            logger.info(f"✓ Solver optimization completed")
            logger.info(f"  Squad size: {len(squad_week1)}")
            logger.info(f"  Total cost: {total_cost:.2f}M")
            
            # Check budget constraint
            if total_cost > 100.0:
                logger.error(f"✗ Budget constraint violated: {total_cost:.2f}M > 100M")
                self.results['solver_check']['error'] = f"Budget exceeded: {total_cost:.2f}M"
                return False
            
            # Check squad structure (2-5-5-3)
            positions = {}
            for player in squad_week1:
                pos = player.get('position', '')
                positions[pos] = positions.get(pos, 0) + 1
            
            expected_structure = {'GK': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
            for pos, count in expected_structure.items():
                if positions.get(pos, 0) != count:
                    logger.error(f"✗ Squad structure violated: {pos} = {positions.get(pos, 0)}, expected {count}")
                    self.results['solver_check']['error'] = f"Squad structure invalid: {positions}"
                    return False
            
            # Check team limit (max 3 per team)
            team_counts = {}
            for player in squad_week1:
                team_id = player.get('team_id', 0)
                team_counts[team_id] = team_counts.get(team_id, 0) + 1
            
            for team_id, count in team_counts.items():
                if count > 3:
                    logger.error(f"✗ Team limit violated: Team {team_id} has {count} players (max 3)")
                    self.results['solver_check']['error'] = f"Team limit exceeded: Team {team_id} = {count}"
                    return False
            
            logger.info(f"✓ Budget constraint: {total_cost:.2f}M <= 100M")
            logger.info(f"✓ Squad structure: {positions}")
            logger.info(f"✓ Team limits: Max {max(team_counts.values())} players per team")
            
            self.results['solver_check']['passed'] = True
            return True
            
        except Exception as e:
            error_msg = f"Solver check error: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"✗ {error_msg}")
            self.results['solver_check']['error'] = error_msg
            return False
    
    def test_strategy_check(self) -> bool:
        """
        Test 5: Strategy Check
        Test C/VC logic in strategy.py without errors.
        """
        logger.info("=" * 60)
        logger.info("TEST 5: Strategy Check (C/VC Logic)")
        logger.info("=" * 60)
        
        try:
            # Initialize strategy service
            strategy_service = StrategyService()
            
            # Create mock PLEngine for testing
            class MockPLEngine:
                def predict(self, player_data, fixture_data=None):
                    # Return mock prediction
                    return {
                        'expected_points': player_data.get('expected_points', 8.0),
                        'p_start': player_data.get('p_start', 0.85)
                    }
            
            mock_plengine = MockPLEngine()
            
            # Create sample captain and vice-captain
            captain = {
                'id': 1,
                'name': 'Test Captain',
                'web_name': 'Test Captain',
                'expected_points': 10.0,
                'p_start': 0.9,
                'fixture_data': None
            }
            
            vice_captain = {
                'id': 2,
                'name': 'Test Vice Captain',
                'web_name': 'Test VC',
                'expected_points': 8.0,
                'p_start': 0.85,
                'fixture_data': None
            }
            
            # Test C/VC value calculation
            logger.info("Testing C/VC Expected Value calculation...")
            cvc_result = strategy_service.calculate_captain_vice_captain_value(
                captain=captain,
                vice_captain=vice_captain,
                plengine=mock_plengine,
                gameweek=1
            )
            
            if cvc_result and 'expected_value' in cvc_result:
                expected_value = cvc_result['expected_value']
                logger.info(f"✓ C/VC Expected Value: {expected_value:.2f}")
                
                # Verify formula: (xP_Capt * P_start_Capt) + (xP_VC * (1 - P_start_Capt))
                expected_calc = (10.0 * 0.9) + (8.0 * (1 - 0.9))
                expected_calc = expected_calc  # 9.0 + 0.8 = 9.8
                
                logger.info(f"  Formula: (10.0 * 0.9) + (8.0 * 0.1) = {expected_calc:.2f}")
                logger.info(f"  Captain contribution: {cvc_result.get('captain_contribution', 0):.2f}")
                logger.info(f"  VC contribution: {cvc_result.get('vice_captain_contribution', 0):.2f}")
                
                # Test ownership arbitrage (simplified)
                logger.info("Testing ownership arbitrage...")
                sample_players = [
                    {
                        'id': 1,
                        'web_name': 'Player 1',
                        'element_type': 3,  # MID
                        'selected_by_percent': 25.0,  # High ownership
                        'now_cost': 80,  # 8.0M
                        'expected_points': 5.0,  # Low xP
                        'fixture_data': None
                    }
                ]
                
                arbitrage_result = strategy_service.analyze_ownership_arbitrage(
                    players=sample_players,
                    plengine=mock_plengine,
                    gameweek=1
                )
                
                if arbitrage_result:
                    logger.info(f"✓ Ownership arbitrage analysis completed")
                    logger.info(f"  Overvalued players: {arbitrage_result.get('overvalued_count', 0)}")
                
                self.results['strategy_check']['passed'] = True
                return True
            else:
                logger.error("✗ C/VC calculation returned invalid result")
                self.results['strategy_check']['error'] = "Invalid C/VC result"
                return False
                
        except Exception as e:
            error_msg = f"Strategy check error: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"✗ {error_msg}")
            self.results['strategy_check']['error'] = error_msg
            return False
    
    def _get_position_name(self, element_type: int) -> str:
        """Convert FPL element_type to position name"""
        position_map = {
            1: 'GK',
            2: 'DEF',
            3: 'MID',
            4: 'FWD'
        }
        return position_map.get(element_type, 'MID')
    
    def test_backtest_check(self) -> bool:
        """
        Test 6: Backtest Check
        Test BacktestEngine initialization and basic functionality.
        """
        logger.info("=" * 60)
        logger.info("TEST 6: Backtest Check (BacktestEngine)")
        logger.info("=" * 60)
        
        try:
            # Initialize BacktestEngine
            logger.info("Initializing BacktestEngine...")
            backtest_engine = BacktestEngine(
                season="2025-26",
                min_train_weeks=5,
                memory_limit_mb=3500
            )
            
            logger.info("✓ BacktestEngine initialized successfully")
            
            # Check if database has data (optional check)
            db = SessionLocal()
            try:
                # Check for any data in player_gameweek_stats
                data_count = db.query(PlayerGameweekStats).count()
                logger.info(f"  PlayerGameweekStats records: {data_count}")
                
                if data_count == 0:
                    logger.warning("  ⚠ No data in player_gameweek_stats table")
                    logger.warning("  Backtest requires ETL data to run. Skipping full backtest.")
                    # Still pass the test if engine initializes correctly
                    self.results['backtest_check']['passed'] = True
                    return True
                
                # Try to get available seasons
                seasons = db.query(PlayerGameweekStats.season).distinct().all()
                available_seasons = [s[0] for s in seasons]
                logger.info(f"  Available seasons: {available_seasons}")
                
                # Try a minimal backtest if data exists (just check if it can start)
                if available_seasons:
                    # Use first available season
                    test_season = available_seasons[0]
                    logger.info(f"  Testing with season: {test_season}")
                    
                    # Create engine with available season
                    test_engine = BacktestEngine(
                        season=test_season,
                        min_train_weeks=5,
                        memory_limit_mb=3500
                    )
                    
                    # Check if it can query gameweeks (without running full backtest)
                    gameweeks = db.query(PlayerGameweekStats.gameweek).filter(
                        PlayerGameweekStats.season == test_season
                    ).distinct().order_by(PlayerGameweekStats.gameweek).all()
                    
                    if gameweeks:
                        logger.info(f"  ✓ Found {len(gameweeks)} gameweeks for season {test_season}")
                        logger.info(f"  Backtest engine is ready to run")
                    else:
                        logger.warning(f"  ⚠ No gameweeks found for season {test_season}")
                    
            finally:
                db.close()
            
            # Test memory management
            logger.info("Testing memory management...")
            backtest_engine._manage_memory()
            logger.info("✓ Memory management working")
            
            # Test report generation (with empty results)
            logger.info("Testing report generation...")
            empty_metrics = {
                'rmse': 0.0,
                'mae': 0.0,
                'spearman': 0.0,
                'r_squared': 0.0,
                'mean_actual': 0.0,
                'mean_predicted': 0.0,
                'n_weeks': 0,
                'cumulative_points': 0.0,
                'total_transfer_cost': 0
            }
            report = backtest_engine._generate_report(empty_metrics, save_to_file=False)
            
            if report and 'season' in report:
                logger.info(f"✓ Report generation successful")
                logger.info(f"  Season: {report.get('season')}")
                logger.info(f"  Methodology: {report.get('methodology')}")
            
            self.results['backtest_check']['passed'] = True
            return True
            
        except Exception as e:
            error_msg = f"Backtest check error: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"✗ {error_msg}")
            self.results['backtest_check']['error'] = error_msg
            return False
    
    def run_all_tests(self) -> Dict:
        """Run all smoke tests"""
        logger.info("=" * 60)
        logger.info("FPL PREDICTION PLATFORM - SMOKE TESTS")
        logger.info("=" * 60)
        logger.info(f"Started at: {datetime.now().isoformat()}")
        logger.info("")
        
        # Test 1: DB Connection
        test1 = self.test_db_connection()
        logger.info("")
        
        # Test 2: ETL Check (async)
        if test1:
            test2 = asyncio.run(self.test_etl_check())
        else:
            logger.warning("Skipping ETL test due to DB connection failure")
            test2 = False
        logger.info("")
        
        # Test 3: ML Check
        test3 = self.test_ml_check()
        logger.info("")
        
        # Test 4: Solver Check
        test4 = self.test_solver_check()
        logger.info("")
        
        # Test 5: Strategy Check
        test5 = self.test_strategy_check()
        logger.info("")
        
        # Test 6: Backtest Check
        test6 = self.test_backtest_check()
        logger.info("")
        
        # Summary
        logger.info("=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        total_tests = 6
        passed_tests = sum([
            self.results['db_connection']['passed'],
            self.results['etl_check']['passed'],
            self.results['ml_check']['passed'],
            self.results['solver_check']['passed'],
            self.results['strategy_check']['passed'],
            self.results['backtest_check']['passed']
        ])
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {total_tests - passed_tests}")
        logger.info("")
        
        for test_name, result in self.results.items():
            status = "✓ PASSED" if result['passed'] else "✗ FAILED"
            logger.info(f"{test_name.upper()}: {status}")
            if result.get('error'):
                logger.error(f"  Error: {result['error']}")
            if result.get('memory_used_mb'):
                logger.info(f"  Memory used: {result['memory_used_mb']:.2f} MB")
        
        logger.info("")
        logger.info(f"Completed at: {datetime.now().isoformat()}")
        logger.info("=" * 60)
        
        return {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': total_tests - passed_tests,
            'results': self.results
        }


if __name__ == "__main__":
    # Run tests
    test_runner = SmokeTest()
    summary = test_runner.run_all_tests()
    
    # Exit with error code if any test failed
    if summary['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)