import asyncio
import asyncpg
from app.auth import hash_password
async def create_superadmin():
    conn = await asyncpg.connect("postgresql://postgres:gfhjkm@localhost/wardrobe_db")
    email = "admin@stylist.ai"
    password = hash_password("admin_pidoras")

    await conn.execute("""
        INSERT INTO users (email, hashed_password, is_admin)
        VALUES ($1, $2, TRUE)
        ON CONFLICT (email) DO NOTHING
    """, email, password)

    await conn.close()
asyncio.run(create_superadmin())