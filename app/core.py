import io
from typing import Optional
from datetime import datetime, timezone

import torch
import clip
import numpy as np
from typing import Dict, Any

from io import BytesIO
from PIL import Image
try:
    from rembg import remove as rembg_remove, new_session
    _HAS_REMBG = True
    # Создаем сессию один раз глобально, чтобы не грузить её при каждом фото
    # u2net_cloth_seg — модель для сегментации одежды
    cloth_session = new_session("u2net_cloth_seg")
except Exception as e:
    print(f"Ошибка инициализации rembg: {e}")
    _HAS_REMBG = False
    cloth_session = None
# Константы (копия из af.py, используемые в логике)
CLOTHING_CATEGORIES = [
    "t-shirt", "top", "shirt", "blouse", "hoodie", "sweater", "cardigan",
    "jeans", "trousers", "leggings", "skirt", "shorts",
    "dress", "jumpsuit",
    "jacket", "suit jacket", "coat", "raincoat",
    "sneakers", "boots", "heels", "sandals",
    "hat", "scarf", "bag", "accessories"
]

CATEGORY_MAP = {
    "t-shirt": "футболка", "top": "топ", "shirt": "рубашка", "blouse": "блузка",
    "hoodie": "худи", "sweater": "свитер", "cardigan": "кардиган",
    "jeans": "джинсы", "trousers": "брюки", "leggings": "леггинсы", "skirt": "юбка", "shorts": "шорты",
    "dress": "платье", "jumpsuit": "комбинезон",
    "jacket": "куртка", "suit jacket": "пиджак", "coat": "пальто", "raincoat": "плащ",
    "sneakers": "кроссовки", "boots": "ботинки", "heels": "туфли/каблуки", "sandals": "сандалии",
    "hat": "шапка/шляпа", "scarf": "шарф", "bag": "сумка",
    "accessories": "аксессуары"
}
DEFAULT_REQUIRED_ALIASES = {
    "top_base": ["футболка","топ","рубашка","блузка","худи","свитер","t-shirt", "top", "shirt", "blouse", "hoodie", "sweater"],
    "bottom": ["низ","брюки","джинсы","юбка","pants","jeans","skirt","shorts","trousers"],
    "shoes": ["обувь","кроссовки","ботинки","туфли","sneaker","shoes","boots","loafer"]
}
CATEGORY_GROUPS = {
    "all": {"label": "Все вещи", "items": None},
    "outer": {"label": "Верхняя одежда", "items": ["coat", "jacket", "raincoat"]},
    "tops": {"label": "Верх", "items": ["t-shirt", "top", "shirt", "blouse", "hoodie", "sweater", "cardigan", "suit jacket"]},
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


# Параметры — можно настраивать
# app/core.py
import random
import numpy as np
import math
from typing import Tuple, List, Dict, Any

# веса и параметры
W_SIM = 1.0
W_COLOR = 0.6
W_MATERIAL = 0.8
DEFAULT_GROUP_SIZE = 6
DEFAULT_CANDIDATES = 120

def _to_vec_from_db(emb_bytes: bytes) -> np.ndarray:
    if emb_bytes is None:
        return None
    try:
        return np.frombuffer(emb_bytes, dtype=np.float32)
    except Exception:
        try:
            return np.frombuffer(memoryview(emb_bytes), dtype=np.float32)
        except Exception:
            return None

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
def _norm_cat_value(row) -> str:
    # пробуем manual_category -> category_en -> category_ru
    v = (row.get("manual_category_en") or row.get("manual_category_ru") or row.get("category_en") or row.get("category_ru") or "")
    return v.strip().lower()

def _matches_aliases(cat_norm: str, aliases: List[str]) -> bool:
    if not cat_norm:
        return False
    for a in aliases:
        if a in cat_norm:
            return True
    return False

async def generate_capsule_items_for_user(pool, user_id:int = 0,
                                          group_size:int = DEFAULT_GROUP_SIZE,
                                          candidates_per_group:int = DEFAULT_CANDIDATES,
                                          required_aliases:Dict[str,List[str]] = None,
                                          max_retries:int = 5) -> Tuple[List[Dict[str,Any]], float, List[str]]:
    """
    Возвращает (items, avg_pair_sim, warnings)
    items: list of dict с полями id, file_id, file_path, name, color_ru, manual_color, manual_material
    """
    if required_aliases is None:
        required_aliases = DEFAULT_REQUIRED_ALIASES

    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, file_id, name, color_ru, color_en,
                   manual_color, manual_material,
                   category_en, category_ru, manual_category_en, manual_category_ru,
                   emb
            FROM wardrobe
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2
        """, user_id, candidates_per_group)

    if not rows:
        return [], 0.0, ["Нет вещей в гардеробе."]

    # подготовка кандидатов
    candidates = []
    for r in rows:
        vec = _to_vec_from_db(r["emb"])
        candidates.append({
            "id": r["id"],
            "file_id": r["file_id"],
            "name": r["name"],
            "color_ru": r.get("color_ru"),
            "color_en": r.get("color_en"),
            "manual_color": r.get("manual_color"),
            "manual_material": r.get("manual_material"),
            "category_norm": _norm_cat_value(r),
            "emb": vec
        })

    # группируем кандидатов по required категориям (если совпадает alias)
    category_buckets = {k: [] for k in required_aliases.keys()}
    others = []
    for c in candidates:
        matched = False
        for k, aliases in required_aliases.items():
            if _matches_aliases(c["category_norm"], aliases):
                category_buckets[k].append(c)
                matched = True
                break
        if not matched:
            others.append(c)

    warnings = []
    # пробуем собрать одну вещь из каждой обязательной категории (если возможно)
    required_selected = []
    for k, bucket in category_buckets.items():
        if bucket:
            # выбираем случайный элемент из бакета (или можно попытаться выбрать самый подходящий по emb)
            chosen = random.choice(bucket)
            required_selected.append(chosen)
        else:
            warnings.append(f"Нет предметов в категории {k}")

    # убираем выбранные из общего пула
    selected_ids = {s["id"] for s in required_selected}
    pool_remaining = [c for c in candidates if c["id"] not in selected_ids]

    # если требуемых уже >= group_size — усечём
    if len(required_selected) >= group_size:
        final_selected = required_selected[:group_size]
        # calc avg simil
        sims = []
        for i in range(len(final_selected)):
            for j in range(i+1, len(final_selected)):
                sims.append(_cosine(final_selected[i]["emb"], final_selected[j]["emb"]))
        avg_pair_sim = (sum(sims)/len(sims)) if sims else 0.0
        out_items = [{
            "id": s["id"],
            "file_id": s["file_id"],
            "name": s.get("name"),
            "color_ru": s.get("color_ru"),
            "manual_color": s.get("manual_color"),
            "manual_material": s.get("manual_material"),
            "file_path": f"/static/uploads/{s['file_id']}" if s.get("file_id") else None
        } for s in final_selected]
        return out_items, float(avg_pair_sim), warnings

    # теперь greedy-дополнение до group_size (с несколькими попытками)
    for attempt in range(max_retries):
        # если есть seed — используй список required_selected как начальный selected
        selected = list(required_selected)
        remaining = pool_remaining.copy()
        # для разнообразия перемешиваем remaining
        random.shuffle(remaining)

        while len(selected) < group_size and remaining:
            best_score = -math.inf
            best_idx = None
            for idx, cand in enumerate(remaining):
                sims = [ _cosine(cand["emb"], s["emb"]) for s in selected if s["emb"] is not None ]
                avg_sim = (sum(sims)/len(sims)) if sims else 0.0

                cand_color = (cand.get("manual_color") or cand.get("color_en") or cand.get("color_ru") or "").strip().lower()
                sel_colors = [(s.get("manual_color") or s.get("color_en") or s.get("color_ru") or "").strip().lower() for s in selected]
                color_match = 1.0 if cand_color and any(cand_color == sc for sc in sel_colors) else 0.0

                cand_mat = (cand.get("manual_material") or "").strip().lower()
                sel_mats = [(s.get("manual_material") or "").strip().lower() for s in selected]
                material_match = 1.0 if cand_mat and any(cand_mat == sm for sm in sel_mats) else 0.0

                score = W_SIM * avg_sim + W_COLOR * color_match + W_MATERIAL * material_match
                score += random.uniform(-0.01, 0.01)

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                break
            selected.append(remaining.pop(best_idx))

        # посчитаем подпись и уникальность
        ids = [s["id"] for s in selected]
        signature = ",".join(str(x) for x in sorted(ids))

        async with pool.acquire() as conn:
            exists = await conn.fetchval("SELECT 1 FROM capsules WHERE user_id=$1 AND signature=$2", user_id, signature)
            if exists:
                # пытаться ещё раз с другим shuffle
                continue

        # вычисляем avg pairwise similarity
        sims = []
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                sims.append(_cosine(selected[i]["emb"], selected[j]["emb"]))
        avg_pair_sim = (sum(sims)/len(sims)) if sims else 0.0

        out_items = [{
            "id": s["id"],
            "file_id": s["file_id"],
            "name": s.get("name"),
            "color_ru": s.get("color_ru"),
            "manual_color": s.get("manual_color"),
            "manual_material": s.get("manual_material"),
            "file_path": f"/static/uploads/{s['file_id']}" if s.get("file_id") else None
        } for s in selected]

        return out_items, float(avg_pair_sim), warnings

    # если не нашлось уникальной комбинации за max_retries — вернуть последний вариант (даже если duplicate)
    sims = []
    for i in range(len(selected)):
        for j in range(i+1, len(selected)):
            sims.append(_cosine(selected[i]["emb"], selected[j]["emb"]))
    avg_pair_sim = (sum(sims)/len(sims)) if sims else 0.0
    out_items = [{
        "id": s["id"],
        "file_id": s["file_id"],
        "name": s.get("name"),
        "color_ru": s.get("color_ru"),
        "manual_color": s.get("manual_color"),
        "manual_material": s.get("manual_material"),
        "file_path": f"/static/uploads/{s['file_id']}" if s.get("file_id") else None
    } for s in selected]
    return out_items, float(avg_pair_sim), warnings


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

import os

# Prefer rembg (U2-Net). Install: pip install rembg
try:
    from rembg import remove as rembg_remove
    _HAS_REMBG = True
except Exception:
    _HAS_REMBG = False

# fallback: OpenCV GrabCut
try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def remove_bg_rembg(input_bytes: bytes) -> bytes:
    if not _HAS_REMBG or cloth_session is None:
        raise RuntimeError("rembg или модель u2net_cloth_seg не доступны")
    try:
        out_bytes = rembg_remove(input_bytes, session=cloth_session)
        return out_bytes
    except Exception as e:
        print(f"Ошибка при обработке фото в rembg: {e}")
        raise e
def remove_bg_grabcut(input_bytes: bytes) -> bytes:
    """
    Fallback: simple GrabCut mask, approximate background removal.
    Returns PNG bytes with alpha.
    """
    if not _HAS_CV2:
        raise RuntimeError("opencv not installed")
    # load to numpy via PIL
    img = Image.open(BytesIO(input_bytes)).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    # initial mask: probable background border
    mask = np.zeros((h, w), np.uint8)
    rect = (max(1, int(w*0.05)), max(1, int(h*0.05)), max(2, int(w*0.9)), max(2, int(h*0.9)))
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(arr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    # mask where sure/probable foreground
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    alpha = (mask2 * 255).astype('uint8')

    rgba = np.dstack([arr, alpha])
    out_img = Image.fromarray(rgba)
    buf = BytesIO()
    out_img.save(buf, format='PNG')
    return buf.getvalue()

def crop_center_and_pad_transparent(png_bytes: bytes, target_size=(600,600)) -> bytes:
    """
    Optional: центр-кроп и паддинг до target_size, сохраняя прозрачность.
    Возвращает PNG bytes.
    """
    img = Image.open(BytesIO(png_bytes)).convert("RGBA")
    w, h = img.size
    # center-crop to fit (preserve aspect)
    aspect_target = target_size[0] / target_size[1]
    aspect_img = w / h
    if aspect_img > aspect_target:
        # crop width
        new_w = int(h * aspect_target)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        # crop height
        new_h = int(w / aspect_target)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    # resize to target (preserve transparency)
    img = img.resize(target_size, Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def remove_background_auto(input_bytes: bytes, use_rembg_if_available=True, pad_size=(600, 600)) -> bytes:
    png_bytes = None

    if use_rembg_if_available and _HAS_REMBG:
        try:
            png_bytes = remove_bg_rembg(input_bytes)
        except Exception as e:
            print(f"Rembg failed, trying fallback: {e}")
            png_bytes = None

    # Если rembg не сработал, пробуем GrabCut (но он оставит тело!)
    if png_bytes is None and _HAS_CV2:
        try:
            png_bytes = remove_bg_grabcut(input_bytes)
        except Exception:
            png_bytes = None

    if png_bytes is None:
        raise RuntimeError("Не удалось удалить фон ни одним способом")

    # Центрируем и добавляем поля, чтобы одежда не прилипала к краям
    if pad_size:
        try:
            png_bytes = crop_center_and_pad_transparent(png_bytes, target_size=pad_size)
        except Exception:
            pass

    return png_bytes