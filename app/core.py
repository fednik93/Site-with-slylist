import io
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone

import numpy as np
import torch
import clip
from PIL import Image

# Константы (копия из af.py, используемые в логике)
CLOTHING_CATEGORIES = [
    "t-shirt", "top", "shirt", "blouse", "hoodie", "sweater", "cardigan",
    "jeans", "trousers", "leggings", "skirt", "shorts",
    "dress", "jumpsuit",
    "jacket", "coat", "raincoat",
    "sneakers", "boots", "heels", "sandals",
    "hat", "scarf", "bag", "accessories"
]

CATEGORY_MAP = {
    "t-shirt": "футболка", "top": "топ", "shirt": "рубашка", "blouse": "блузка",
    "hoodie": "худи", "sweater": "свитер", "cardigan": "кардиган",
    "jeans": "джинсы", "trousers": "брюки", "leggings": "леггинсы", "skirt": "юбка", "shorts": "шорты",
    "dress": "платье", "jumpsuit": "комбинезон",
    "jacket": "куртка", "coat": "пальто", "raincoat": "плащ",
    "sneakers": "кроссовки", "boots": "ботинки", "heels": "туфли/каблуки", "sandals": "сандалии",
    "hat": "шапка/шляпа", "scarf": "шарф", "bag": "сумка",
    "accessories": "аксессуары"
}

CATEGORY_GROUPS = {
    "all": {"label": "Все вещи", "items": None},
    "outer": {"label": "Верхняя одежда", "items": ["coat", "jacket", "raincoat"]},
    "tops": {"label": "Верх", "items": ["t-shirt", "top", "shirt", "blouse", "hoodie", "sweater", "cardigan"]},
    "bottoms": {"label": "Низ", "items": ["jeans", "trousers", "leggings", "shorts", "skirt"]},
    "dresses": {"label": "Платья/комбинезоны", "items": ["dress", "jumpsuit"]},
    "shoes": {"label": "Обувь", "items": ["sneakers", "boots", "heels", "sandals"]},
    "accessories": {"label": "Аксессуары", "items": ["hat", "scarf", "bag"]}
}

COLOR_LABELS = ["white","black","gray","red","orange","yellow","green","blue","purple","pink","brown","beige","maroon","olive"]
COLOR_MAP = {"white":"белый","black":"чёрный","gray":"серый","red":"красный","orange":"оранжевый","yellow":"жёлтый",
             "green":"зелёный","blue":"синий","purple":"фиолетовый","pink":"розовый","brown":"коричневый","beige":"бежевый",
             "maroon":"бордовый","olive":"оливковый"}

PAGE_SIZE = 10

# Модель и препроцесс — инициализируются через init_clip()
_model = None
_preprocess = None
_device = None

def init_clip(device: Optional[str] = None, model_name: str = "ViT-B/32", jit: bool = False):
    """
    Инициализация CLIP. Вызывать при старте приложения (или лениво перед первым использованием).
    device: "cpu" / "cuda" / None
    """
    global _model, _preprocess, _device
    if _model is not None:
        return
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    _device = device
    _model, _preprocess = clip.load(model_name, device=device, jit=jit)
    # оставляем модель в глобале

def image_to_embedding(pil_image: Image.Image) -> np.ndarray:
    """
    Преобразует PIL.Image в нормализованный вектор (numpy float32).
    Возвращает None при ошибке.
    """
    if _model is None or _preprocess is None:
        init_clip()
    image_input = _preprocess(pil_image).unsqueeze(0).to(_device)
    with torch.no_grad():
        emb = _model.encode_image(image_input)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype(np.float32).squeeze(0)

def analyze_image_bytes(b: bytes) -> Dict[str, Any]:
    """
    Удобная обёртка: получает raw bytes изображения, возвращает
    { 'emb': np.ndarray, 'category_en', 'category_ru', 'category_conf', 'color_en', 'color_ru', 'color_conf' }
    """
    bio = io.BytesIO(b)
    pil_image = Image.open(bio).convert("RGB")
    emb = image_to_embedding(pil_image)

    # logits (используем ту же логику, что в af.py)
    image_input = _preprocess(pil_image).unsqueeze(0).to(_device)
    cat_logits = clip_infer_logits(image_input); cat_probs = torch.softmax(cat_logits, dim=0)
    top_idx = int(torch.argmax(cat_probs).item())
    top_cat_en = CLOTHING_CATEGORIES[top_idx]
    top_cat_ru = CATEGORY_MAP.get(top_cat_en, top_cat_en)
    top_cat_conf = float(cat_probs[top_idx].item())

    color_logits = clip_color_logits(image_input); color_probs = torch.softmax(color_logits, dim=0)
    top_color_vals = torch.topk(color_probs, k=1)
    top_color_en = COLOR_LABELS[int(top_color_vals.indices[0])]
    top_color_ru = COLOR_MAP.get(top_color_en, top_color_en)
    top_color_conf = float(top_color_vals.values[0])

    return {
        "emb": emb,
        "category_en": top_cat_en,
        "category_ru": top_cat_ru,
        "category_conf": top_cat_conf,
        "color_en": top_color_en,
        "color_ru": top_color_ru,
        "color_conf": top_color_conf
    }

# ---------------- CLIP helpers (копия логики из af.py) ----------------
def clip_infer_logits(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Оригинальная логика расчёта логитов по категориям (взято из af.py).
    Возвращает тензор логитов по CLOTHING_CATEGORIES.
    """
    if _model is None:
        init_clip()
    with torch.no_grad():
        text_cat = [f"a photo of a {c}" for c in CLOTHING_CATEGORIES]
        text_tokens = clip.tokenize(text_cat).to(_device)
        image_features = _model.encode_image(image_tensor)
        text_features = _model.encode_text(text_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = _model.logit_scale.exp().to(_device)
        cat_logits = (image_features @ text_features.t()).squeeze(0) * logit_scale
    return cat_logits.cpu()

def clip_color_logits(image_tensor: torch.Tensor) -> torch.Tensor:
    if _model is None:
        init_clip()
    with torch.no_grad():
        text_colors = [f"the color is {c}" for c in COLOR_LABELS]
        color_tokens = clip.tokenize(text_colors).to(_device)
        image_features = _model.encode_image(image_tensor)
        color_features = _model.encode_text(color_tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        color_features = color_features / color_features.norm(dim=-1, keepdim=True)
        logit_scale = _model.logit_scale.exp().to(_device)
        color_logits = (image_features @ color_features.t()).squeeze(0) * logit_scale
    return color_logits.cpu()

# ---------------- Утилиты ----------------
def to_vector_from_bytes(b: Optional[bytes]) -> Optional[np.ndarray]:
    if b is None:
        return None
    return np.frombuffer(b, dtype=np.float32)

def cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return -1.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

# ---------------- Capsule generation (async, требует db_pool) ----------------
# Эта функция напрямую адаптирована из af.py: она ожидает asyncpg pool (db_pool)
async def generate_capsule_items_for_user(db_pool, user_id: int, candidates_per_group: int = 25) -> Tuple[List[Dict[str, Any]], float]:
    """
    Возвращает (selected_items, avg_pair_sim).
    selected_items — список словарей с ключами id,file_id,name,color_ru,category_en,emb_vec.
    Требует: db_pool — asyncpg pool (как в твоем проекте).
    """
    groups = {
        "tops": CATEGORY_GROUPS["tops"]["items"],
        "bottoms": CATEGORY_GROUPS["bottoms"]["items"],
        "dresses": CATEGORY_GROUPS["dresses"]["items"],
        "outer": CATEGORY_GROUPS["outer"]["items"],
        "shoes": CATEGORY_GROUPS["shoes"]["items"],
        "accessories": CATEGORY_GROUPS["accessories"]["items"]
    }

    async with db_pool.acquire() as conn:
        async def fetch_candidates(categories):
            if not categories:
                return []
            rows = await conn.fetch(
                "SELECT id, file_id, name, color_ru, category_en, emb FROM wardrobe WHERE user_id=$1 AND category_en = ANY($2::text[]) AND emb IS NOT NULL LIMIT $3",
                user_id, categories, candidates_per_group
            )
            items = []
            for r in rows:
                vec = to_vector_from_bytes(r['emb'])
                if vec is None:
                    continue
                items.append({
                    "id": r['id'],
                    "file_id": r['file_id'],
                    "name": r['name'] or "",
                    "color_ru": r['color_ru'] or "",
                    "category_en": r['category_en'] or "",
                    "emb_vec": vec
                })
            return items

        candidates = {k: await fetch_candidates(v) for k, v in groups.items()}

    selected: List[Dict[str, Any]] = []

    # логика выбора (копия из af.py)
    if candidates.get("dresses"):
        selected.append(candidates["dresses"][0])
    else:
        tops = candidates.get("tops", [])
        bottoms = candidates.get("bottoms", [])
        best_pair = (None, None, -999.0)
        for t in tops:
            for b in bottoms:
                s = cosine_sim(t['emb_vec'], b['emb_vec'])
                if s > best_pair[2]:
                    best_pair = (t, b, s)
        if best_pair[0] and best_pair[1]:
            selected.append(best_pair[0]); selected.append(best_pair[1])
        else:
            if tops:
                selected.append(tops[0])
            elif bottoms:
                selected.append(bottoms[0])

    def centroid(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
        if not vectors:
            return None
        arr = np.vstack(vectors)
        c = np.mean(arr, axis=0)
        norm = np.linalg.norm(c) + 1e-8
        return c / norm

    SIM_THRESHOLD = 0.18

    for slot in ("outer", "shoes", "accessories"):
        pool = candidates.get(slot, []) or []
        if not pool:
            continue
        cent = centroid([s['emb_vec'] for s in selected]) if selected else None
        best_cand = None; best_score = -999.0
        for cand in pool:
            score = cosine_sim(cand['emb_vec'], cent) if cent is not None else 0.0
            if score > best_score:
                best_score = score; best_cand = cand
        if best_cand and best_score >= SIM_THRESHOLD:
            selected.append(best_cand)

    if len(selected) < 2:
        for k in ("tops","bottoms","dresses","outer","shoes","accessories"):
            if candidates.get(k):
                selected.append(candidates[k][0])
                if len(selected) >= 2: break

    avg_pair_sim = 0.0
    if len(selected) >= 2:
        sims = []
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                sims.append(cosine_sim(selected[i]['emb_vec'], selected[j]['emb_vec']))
        avg_pair_sim = float(np.mean(sims)) if sims else 0.0

    return selected, avg_pair_sim

# ---------------- Convenience: сохранить капсулу в БД ----------------
async def save_capsule(db_pool, user_id: int, name: str, item_ids: List[int], thumbnail_file_id: Optional[str] = None) -> int:
    """
    Сохраняет капсулу в таблицу capsules и возвращает id новой капсулы.
    """
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow("""
            INSERT INTO capsules (user_id, name, item_ids, thumbnail_file_id, created_at)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
        """, user_id, name, item_ids, thumbnail_file_id, datetime.now(timezone.utc))
    return int(row['id'])

# ---------------- Пояснения ----------------
# 1) Этот модуль вынесёт всю ML-логику отдельно от Telegram-кода.
# 2) Функции analyze_image_bytes и image_to_embedding удобно вызывать из FastAPI-эндпоинта при загрузке файла.
# 3) Функция generate_capsule_items_for_user требует asyncpg pool (db_pool) — передавайте тот же объект,
#    что у вас сейчас в боте (в af.py он называется db_pool). Логика почти дословно перенесена из af.py.
#
# Пример использования (FastAPI sync endpoint):
#   from app.core import analyze_image_bytes
#   @app.post("/analyze")
#   async def analyze(file: UploadFile = File(...)):
#       b = await file.read()
#       info = analyze_image_bytes(b)
#       return info
#
# Пример использования (генерация капсулы, async):
#   from app.core import generate_capsule_items_for_user
#   items, sim = await generate_capsule_items_for_user(db_pool, user_id)
#
# Код генерации и инференса основан на логике из твоего af.py. См. оригинал (CLIP helpers + generation). :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}
