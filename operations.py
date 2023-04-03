from fastapi import FastAPI

from connect_db import pool
import pymysql
app = FastAPI()
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    async with pool.acquire() as conn:
        async with conn.cursor(pymysql.cursors.DictCursor) as cur:
            await cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            return await cur.fetchone()