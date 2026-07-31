import asyncio

import asyncpg


async def main() -> None:
    conn = await asyncpg.connect(
        host="127.0.0.1",
        port=55432,
        user="aegis",
        password="aegis",
        database="aegis",
    )
    print("connect_ok", await conn.fetchval("SELECT 1"))
    await conn.close()


asyncio.run(main())
