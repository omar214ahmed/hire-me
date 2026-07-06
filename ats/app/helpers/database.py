import asyncio
import json

import asyncpg

from helpers.config import get_settings

settings = get_settings()

_pool = None
_pool_loop_id = None


async def get_pool():
    global _pool, _pool_loop_id
    current_loop_id = id(asyncio.get_running_loop())
    if _pool is None or _pool_loop_id != current_loop_id:
        _pool = await asyncpg.create_pool(
            dsn=settings.DATABASE_URL,
            min_size=2,
            max_size=10,
        )
        _pool_loop_id = current_loop_id
    return _pool


async def close_pool():
    global _pool, _pool_loop_id
    if _pool is not None:
        current_loop_id = id(asyncio.get_running_loop())
        if _pool_loop_id == current_loop_id:
            await _pool.close()
        _pool = None
        _pool_loop_id = None