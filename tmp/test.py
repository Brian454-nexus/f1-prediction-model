import sys
sys.path.append('.')
import asyncio
from api.main import generate_prediction

async def test():
    try:
        res = await generate_prediction(round_id=2, circuit_name='China', total_laps=56, race_utc='2026-04-19T07:00:00Z')
        print('SUCCESS')
    except Exception as e:
        import traceback
        traceback.print_exc()

asyncio.run(test())
