# app/main.py (верх файла)
import os
import shutil
from typing import Optional, List
from datetime import datetime, timezone

import asyncpg
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.auth import hash_password, verify_password
from app.core import analyze_image_bytes, init_clip, generate_capsule_items_for_user
APP_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(APP_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Stylist MVP")
templates = Jinja2Templates(directory=os.path.join(APP_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")

SECRET_KEY = os.getenv("SECRET_KEY", "change_this_to_a_random_long_secret")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# DB pool (None if not configured)
db_pool: Optional[asyncpg.pool.Pool] = None
DATABASE_URL = os.getenv("DATABASE_URL")  # установить в окружении если хочешь БД
from typing import Optional

def get_current_user_id(request: Request) -> Optional[int]:
    return request.session.get("user_id")
# асинхронно получить текущего пользователя (если есть)
async def get_current_user(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id, email FROM users WHERE id=$1", user_id)
        if not row:
            return None
        return dict(row)

# удобный рендер: всегда подставляет request и current_user
async def render_template(request: Request, template_name: str, context: dict | None = None):
    if context is None:
        context = {}
    current_user = await get_current_user(request)
    context.setdefault("current_user", current_user)
    context["request"] = request
    return templates.TemplateResponse(template_name, context)

# --- DB init helpers (создаёт нужные таблицы, без миграций) ---
async def init_db_and_migrate(pool: asyncpg.pool.Pool):
    async with pool.acquire() as conn:
        await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email TEXT Unique NOT NULL,
                    hashed_password TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
                );
                """)
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS wardrobe (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            file_id TEXT NOT NULL,
            emb BYTEA,
            name TEXT,
            color_en TEXT,
            color_ru TEXT,
            category_en TEXT,
            category_ru TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            description TEXT DEFAULT ''
        );
        """)
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS capsules (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            item_ids INTEGER[] NOT NULL,
            thumbnail_file_id TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
            description TEXT DEFAULT ''
        );
        """)
async def get_current_user(request: Request):
    """Возвращает dict пользователя или None. Асинхронно читает из БД если нужно."""
    user_id = request.session.get("user_id")
    if not user_id or not db_pool:
        return None
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id, email FROM users WHERE id=$1", int(user_id))
        return dict(row) if row else None

def base_context(request: Request, extra: dict = None):
    """Базовый контекст для шаблонов — request + info о текущем пользователе или None."""
    ctx = {"request": request}
    # current_user добавим синхронно как placeholder; лучше в обработчике подгружать асинхронно
    if extra:
        ctx.update(extra)
    return ctx

# --- Startup / Shutdown ---
@app.on_event("startup")
async def on_startup():
    # CLIP
    init_clip()

    # DB pool (если DATABASE_URL задан)
    global db_pool
    if DATABASE_URL:
        db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
        await init_db_and_migrate(db_pool)

@app.on_event("shutdown")
async def on_shutdown():
    global db_pool
    if db_pool:
        await db_pool.close()

# ---------------- Routes ----------------

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return await render_template(request, "index.html")


# регистрация - GET показывает форму, POST обрабатывает
@app.get("/register", response_class=HTMLResponse)
async def register_get(request: Request):
    return await render_template(request, "register.html")

@app.post("/register", response_class=HTMLResponse)
async def register_post(request: Request, email: str = Form(...), password: str = Form(...)):
    if not db_pool:
        return HTMLResponse("Database not configured", status_code=500)
    # проверяем уникальность email
    async with db_pool.acquire() as conn:
        exists = await conn.fetchval("SELECT 1 FROM users WHERE email=$1", email)
        if exists:
            return await render_template(request, "register.html", {"error": "Этот email уже зарегистрирован."})
        hashed = hash_password(password)
        row = await conn.fetchrow("INSERT INTO users (email, hashed_password, created_at) VALUES ($1, $2, now()) RETURNING id", email, hashed)
        user_id = row["id"]
    # логиним пользователя — кладём в сессию
    request.session["user_id"] = int(user_id)
    return RedirectResponse(url="/wardrobe", status_code=303)


# логин
@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request):
    return await render_template(request, "login.html")

@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request, email: str = Form(...), password: str = Form(...)):
    if not db_pool:
        return HTMLResponse("Database not configured", status_code=500)
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id, hashed_password FROM users WHERE email=$1", email)
    if not row:
        return await render_template(request, "login.html", {"error": "Этот email уже зарегистрирован."})
    if not verify_password(password, row["hashed_password"]):
        return await render_template(request, "login.html", {"error": "Неверный email или пароль."})
    request.session["user_id"] = int(row["id"])
    return RedirectResponse(url="/wardrobe", status_code=303)

# logout
@app.post("/logout")
async def logout(request: Request):
    request.session.pop("user_id", None)
    return RedirectResponse(url="/", status_code=303)


# ---------------- Wardrobe ----------------
@app.get("/wardrobe", response_class=HTMLResponse)
async def view_wardrobe(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse("/login")
    if not db_pool:
        raise HTTPException(status_code=500)
    items = []
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, file_id, name,
                   color_ru, category_ru,
                   manual_category_ru, manual_category_en,
                   manual_color,
                   created_at
            FROM wardrobe
            WHERE user_id=$1
            ORDER BY created_at DESC
        """, user["id"])
        for r in rows:
            it = dict(r)
            it["display_category"] = (
                it.get("manual_category_ru")
                or it.get("manual_category_en")
                or it.get("category_ru")
            )
            it["display_color"] = (
                it.get("manual_color")
                or it.get("color_ru")
            )
            items.append(it)
    return await render_template(request, "wardrobe.html", {"items": items})

@app.post("/wardrobe/add", response_class=HTMLResponse)
async def add_to_wardrobe(request: Request, file: UploadFile = File(...), name: Optional[str] = Form(None)):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse("/login")

    safe_name = f"{user['id']}_{int(datetime.now().timestamp())}_{file.filename.replace('/', '_')}"
    dest_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    with open(dest_path, "rb") as f:
        b = f.read()
    info = analyze_image_bytes(b)

    if db_pool:
        emb = info.get("emb")
        emb_bytes = emb.tobytes() if emb is not None else None
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO wardrobe (user_id, file_id, emb, name, color_en, color_ru, category_en, category_ru, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """, user["id"], safe_name, emb_bytes, name or info.get("category_ru"), info.get("color_en"), info.get("color_ru"),
                 info.get("category_en"), info.get("category_ru"), datetime.now(timezone.utc))
    return RedirectResponse(url="/wardrobe", status_code=303)


@app.get("/wardrobe/item/{item_id}", response_class=HTMLResponse)
async def view_item(request: Request, item_id: int):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse("/login")
    if not db_pool:
        raise HTTPException(status_code=500)
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT id, file_id, name, color_ru, category_ru,
                   manual_category_ru, manual_category_en,
                   manual_color, description, created_at, user_id
            FROM wardrobe
            WHERE id=$1
        """, item_id)
        if not row or row["user_id"] != user["id"]:
            raise HTTPException(status_code=404, detail="Item not found")
        item = dict(row)
        item["display_category"] = (
            item.get("manual_category_ru")
            or item.get("manual_category_en")
            or item.get("category_ru")
        )
        item["display_color"] = (
            item.get("manual_color")
            or item.get("color_ru")
        )
    file_url = f"/static/uploads/{item['file_id']}"
    return await render_template(
        request,
        "item_view.html",
        {"item": item, "file_url": file_url}
    )

@app.post("/wardrobe/item/{item_id}/delete")
async def delete_item(request: Request, item_id: int):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse("/login")
    if not db_pool:
        raise HTTPException(status_code=500, detail="DB not configured")
    async with db_pool.acquire() as conn:
        # проверим право владения
        row = await conn.fetchrow("SELECT file_id, user_id FROM wardrobe WHERE id=$1", item_id)
        if not row or row["user_id"] != user["id"]:
            raise HTTPException(status_code=404)
        # удалить файл с диска (если локально)
        file_name = row["file_id"]
        try:
            os.remove(os.path.join(UPLOAD_DIR, file_name))
        except FileNotFoundError:
            pass
        await conn.execute("DELETE FROM wardrobe WHERE id=$1", item_id)
    return RedirectResponse("/wardrobe", status_code=303)

@app.get("/wardrobe/item/{item_id}/edit", response_class=HTMLResponse)
async def edit_item_get(request: Request, item_id: int):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse("/login")
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id, file_id, name, color_ru, category_ru, category_en, manual_category_ru, manual_color FROM wardrobe WHERE id=$1 AND user_id=$2", item_id, user["id"])
        if not row:
            raise HTTPException(status_code=404)
    item = dict(row)
    return await render_template(request, "item_edit.html", {"item": item})

@app.post("/wardrobe/item/{item_id}/edit")
async def edit_item_post(request: Request, item_id: int, name: Optional[str] = Form(None), manual_category_en: Optional[str] = Form(None), manual_category_ru: Optional[str] = Form(None), manual_color: Optional[str] = Form(None)):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse("/login")
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT id FROM wardrobe WHERE id=$1 AND user_id=$2", item_id, user["id"])
        if not row:
            raise HTTPException(status_code=404)
        await conn.execute("""
            UPDATE wardrobe
            SET name=$1, manual_category_en=$2, manual_category_ru=$3, manual_color=$4
            WHERE id=$5
        """, name, manual_category_en, manual_category_ru, manual_color, item_id)
        # сохраняем в corrections для обучения
        await conn.execute("""
            INSERT INTO label_corrections (wardrobe_id, user_id, corrected_category_en, corrected_category_ru, corrected_color)
            VALUES ($1, $2, $3, $4, $5)
        """, item_id, user["id"], manual_category_en, manual_category_ru, manual_color)
    return RedirectResponse(f"/wardrobe/item/{item_id}", status_code=303)


# ---------------- Capsules ----------------
@app.get("/capsules", response_class=HTMLResponse)
async def view_capsules(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse("/login")
    caps = []
    if db_pool:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, name, item_ids, thumbnail_file_id, created_at FROM capsules WHERE user_id=$1 ORDER BY created_at DESC",
                user["id"]
            )
            caps = [dict(r) for r in rows]
            # можно подготовить thumbnail path
            for c in caps:
                if c.get("thumbnail_file_id"):
                    c["thumbnail_path"] = f"/static/uploads/{c['thumbnail_file_id']}"
    return await render_template(request, "capsules.html", {"capsules": caps})

@app.post("/capsules/generate", response_class=HTMLResponse)
async def generate_capsule_web(request: Request):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse("/login")
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database not configured")
    items, avg_sim = await generate_capsule_items_for_user(db_pool, user_id=user["id"], candidates_per_group=40)
    if not items:
        return await render_template(request, "capsules.html", {
            "capsules": [],
            "message": "Недостаточно вещей для генерации капсулы. Нужно минимум 4–6 предметов."
        })
    # подготовим пути для отображения
    for it in items:
        if "file_id" in it and it.get("file_id"):
            it["file_path"] = f"/static/uploads/{it['file_id']}"
    return await render_template(request, "capsule_generated.html", {"items": items, "avg_sim": avg_sim})

@app.post("/capsules/save", response_class=HTMLResponse)
async def save_capsule_web(request: Request, name: str = Form(...), item_ids: str = Form(...)):

    user = await get_current_user(request)
    if not user:
        return RedirectResponse("/login")

    ids = [int(x) for x in item_ids.split(",") if x.strip()]
    thumb = None

    if db_pool:
        async with db_pool.acquire() as conn:
            if ids:
                row = await conn.fetchrow(
                    "SELECT file_id FROM wardrobe WHERE id=$1 AND user_id=$2",
                    ids[0], user["id"]
                )
                if row:
                    thumb = row["file_id"]

            await conn.execute("""
                INSERT INTO capsules (user_id, name, item_ids, thumbnail_file_id, created_at)
                VALUES ($1, $2, $3, $4, $5)
            """, user["id"], name, ids, thumb, datetime.now(timezone.utc))

    return RedirectResponse(url="/capsules", status_code=303)

@app.get("/capsules/view/{cap_id}", response_class=HTMLResponse)
async def view_capsule(request: Request, cap_id: int):
    user = await get_current_user(request)
    if not user:
        return RedirectResponse("/login")
    cap = None
    items = []
    if db_pool:
        async with db_pool.acquire() as conn:
            caprow = await conn.fetchrow("SELECT id, name, item_ids, thumbnail_file_id, created_at, user_id FROM capsules WHERE id=$1", cap_id)
            if not caprow or caprow["user_id"] != user["id"]:
                raise HTTPException(status_code=404, detail="Capsule not found")
            cap = dict(caprow)
            if cap.get("item_ids"):
                rows = await conn.fetch(
                    "SELECT id, file_id, name, color_ru, category_ru FROM wardrobe WHERE id = ANY($1::int[]) AND user_id=$2",
                    cap["item_ids"], user["id"]
                )
                items = [dict(r) for r in rows]
                for it in items:
                    it["file_path"] = f"/static/uploads/{it.get('file_id')}"
    return await render_template(request, "capsule_view.html", {"cap": cap, "items": items})
# ---------------- API (json) ----------------
@app.post("/api/analyze")
async def analyze_api(file: UploadFile = File(...)):
    b = await file.read()
    info = analyze_image_bytes(b)
    payload = {
        "category_en": info.get("category_en"),
        "category_ru": info.get("category_ru"),
        "category_conf": float(info.get("category_conf", 0)),
        "color_en": info.get("color_en"),
        "color_ru": info.get("color_ru"),
        "color_conf": float(info.get("color_conf", 0))
    }
    return JSONResponse(payload)
