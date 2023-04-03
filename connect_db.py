from fastapi import FastAPI
import pymysql

app = FastAPI()

# MySQL connection settings
MYSQL_HOST = 'localhost'
MYSQL_USER = 'root'
MYSQL_PASSWORD = ''
MYSQL_DB = 'mydatabase'

# MySQL connection pool
pool = None

# Connect to MySQL on startup
@app.on_event("startup")
async def startup():
    global pool
    pool = await create_pool()

# Disconnect from MySQL on shutdown
@app.on_event("shutdown")
async def shutdown():
    if pool is not None:
        pool.close()
        await pool.wait_closed()

# MySQL connection pool factory function
async def create_pool():
    return await pymysql.create_pool(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        db=MYSQL_DB,
        autocommit=True
    )

# Example API endpoint that fetches data from MySQL
