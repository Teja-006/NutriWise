"""
app.py - NutriWise API.

POST /predict
    multipart field: "file"
    Returns detected dishes, estimated grams, and nutrition.

GET /health
    Health check.
"""

import io
import os
import tempfile

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

from estimate_weight import estimate_weights
from food_nutrition import nutrition_for, FIELDS


# ============================================================
# JSON SERIALIZATION HELPER
# ============================================================

def make_json_safe(obj):
    """
    Convert NumPy/Python objects into values that FastAPI
    can safely serialize as JSON.
    """

    # NumPy arrays -> Python lists
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # NumPy scalar values:
    # np.float32 -> float
    # np.int64   -> int
    # np.bool_   -> bool
    if isinstance(obj, np.generic):
        return obj.item()

    # Dictionaries
    if isinstance(obj, dict):
        return {
            key: make_json_safe(value)
            for key, value in obj.items()
        }

    # Lists
    if isinstance(obj, list):
        return [
            make_json_safe(value)
            for value in obj
        ]

    # Tuples
    if isinstance(obj, tuple):
        return [
            make_json_safe(value)
            for value in obj
        ]

    # Normal Python values
    return obj


# ============================================================
# CONFIGURATION
# ============================================================

WEIGHTS = os.getenv(
    "SEG_WEIGHTS",
    "checkpoints/food_seg_multiclass_v3.pth"
)

ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "*"
    ).split(",")
    if origin.strip()
]

MAX_BYTES = 12 * 1024 * 1024


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="NutriWise API"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # 1. Read uploaded image
    # --------------------------------------------------------

    data = file.file.read()

    if len(data) > MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image too large (max 12 MB)."
        )

    # --------------------------------------------------------
    # 2. Open and process image
    # --------------------------------------------------------

    try:
        img = Image.open(
            io.BytesIO(data)
        )

        # Correct orientation from phone photos
        img = ImageOps.exif_transpose(img)

        # Convert to RGB
        img = img.convert("RGB")

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read that file as an image."
        )

    # Keep processing reasonably fast
    img.thumbnail(
        (1600, 1600)
    )

    # --------------------------------------------------------
    # 3. Save temporary image
    # --------------------------------------------------------

    tmp = tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    )

    tmp.close()

    try:

        img.save(
            tmp.name,
            quality=92
        )

        # ----------------------------------------------------
        # 4. Run NutriWise model
        # ----------------------------------------------------

        results, meta = estimate_weights(
            tmp.name,
            WEIGHTS
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {e}"
        )

    finally:

        # Always delete temporary file
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

    # --------------------------------------------------------
    # 5. No plate detected / no result
    # --------------------------------------------------------

    if meta is None:

        response = {
            "items": [],
            "totals": None,
            "total_grams": 0,
            "plate_detected": False
        }

        return make_json_safe(response)

    # --------------------------------------------------------
    # 6. Prepare prediction results
    # --------------------------------------------------------

    items = []

    totals = {
        key: 0.0
        for key in FIELDS
    }

    total_g = 0.0

    counted = False

    # --------------------------------------------------------
    # 7. Process every detected food
    # --------------------------------------------------------

    for r in results:

        # Estimated food weight
        g = r.get("weight_g")

        # Nutrition lookup
        if g is not None:

            n = nutrition_for(
                r["name"],
                g
            )

        else:

            n = None

        # ----------------------------------------------------
        # Add weight
        # ----------------------------------------------------

        if g is not None:

            total_g += float(g)

        # ----------------------------------------------------
        # Add nutrition totals
        # ----------------------------------------------------

        if n:

            counted = True

            for key in FIELDS:

                totals[key] += float(
                    n[key]
                )

        # ----------------------------------------------------
        # Create food item
        # ----------------------------------------------------

        item = {
            "id": int(
                r["class_id"]
            ),

            "name": r["name"],

            "confidence": (
                float(r["confidence"])
                if r.get("confidence") is not None
                else None
            ),

            "area_cm2": float(
                r["area_cm2"]
            ),

            "weight_g": (
                float(g)
                if g is not None
                else None
            ),

            "nutrition": (
                {
                    key: float(value)
                    for key, value in n.items()
                }
                if n
                else None
            ),
        }

        items.append(item)

    # --------------------------------------------------------
    # 8. Build API response
    # --------------------------------------------------------

    response = {

        "items": items,

        "totals": (
            {
                key: round(
                    float(value),
                    1
                )
                for key, value in totals.items()
            }
            if counted
            else None
        ),

        "total_grams": round(
            float(total_g),
            1
        ),

        "plate_detected": bool(
            meta.get(
                "plate_detected",
                False
            )
        ),
    }

    # --------------------------------------------------------
    # 9. Final JSON safety conversion
    # --------------------------------------------------------

    return make_json_safe(
        response
    )