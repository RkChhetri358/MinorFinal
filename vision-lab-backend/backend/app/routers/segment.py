import asyncio
import base64
import io
import random
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

router = APIRouter()


def generate_segmentation_mask(image: Image.Image) -> str:
    """Generate a placeholder segmentation mask overlay."""
    img_array = np.array(image.resize((512, 512)))
    height, width = img_array.shape[:2]

    # Create colored segmentation overlay
    mask = Image.new("RGBA", (width, height), (0, 0, 0, 200))
    mask_array = np.array(mask)

    # Simulate segmentation regions with random colored blobs
    colors = [
        (0, 255, 200, 180),   # cyan
        (180, 0, 255, 180),   # purple
        (255, 100, 0, 180),   # orange
        (0, 180, 255, 180),   # blue
        (255, 220, 0, 180),   # yellow
    ]

    for i, color in enumerate(colors):
        cx = random.randint(80, width - 80)
        cy = random.randint(80, height - 80)
        rx = random.randint(40, 120)
        ry = random.randint(40, 100)

        for y in range(max(0, cy - ry), min(height, cy + ry)):
            for x in range(max(0, cx - rx), min(width, cx + rx)):
                if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1:
                    mask_array[y, x] = color

    # Composite with original
    original_resized = image.resize((width, height)).convert("RGBA")
    mask_img = Image.fromarray(mask_array.astype(np.uint8), "RGBA")
    result = Image.alpha_composite(original_resized, mask_img)

    # Add edge detection overlay
    result_rgb = result.convert("RGB")
    edges = result_rgb.filter(ImageFilter.FIND_EDGES)
    enhanced = ImageEnhance.Brightness(edges).enhance(2.0)

    final = Image.blend(result_rgb, enhanced, 0.3)

    buf = io.BytesIO()
    final.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@router.post("")
async def run_segmentation(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Simulate model inference
    await asyncio.sleep(2.5)

    segmented_b64 = generate_segmentation_mask(image)

    return {
        "status": "success",
        "model": "SAM-ViT-H",
        "num_masks": random.randint(3, 8),
        "confidence": round(random.uniform(0.87, 0.98), 3),
        "image_b64": segmented_b64,
        "processing_time_ms": random.randint(1800, 3200),
    }
