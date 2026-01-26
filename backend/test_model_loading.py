"""Test model loading"""
import asyncio
from app.services.ml_engine import PLEngine

async def test():
    engine = PLEngine()
    await engine.async_load_models()
    print('xMins trained:', engine.xmins_model.is_trained, 'loaded:', engine.xmins_model.is_loaded)
    print('xMins model type:', type(engine.xmins_strategy.model).__name__ if engine.xmins_strategy.model else 'None')
    print('Attack xg trained:', engine.attack_model.xg_trained, 'loaded:', engine.attack_model.is_loaded)
    print('Attack xg_model type:', type(engine.attack_strategy.xg_model).__name__ if engine.attack_strategy.xg_model else 'None')
    result = engine.xmins_strategy.predict({'fpl_id': 1, 'position': 'MID', 'recent_minutes': [90]}, None)
    print('Prediction works:', result)

asyncio.run(test())
