import asyncio
import base64
import io
import random
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image, ImageFilter
import numpy as np

router = APIRouter()


def generate_depth_map(image: Image.Image) -> str:
    """Generate a placeholder depth map with gradient effects."""
    img = image.resize((512, 512)).convert("L")
    img_array = np.array(img, dtype=np.float32)

    # Apply gaussian blur to simulate depth-based smoothing
    blurred = np.array(Image.fromarray(img_array.astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(radius=8)
    ), dtype=np.float32)

    # Create a radial gradient to simulate depth falloff
    height, width = img_array.shape
    cx, cy = width // 2, height // 2
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx**2 + cy**2)
    radial_gradient = 1.0 - (dist_from_center / max_dist) * 0.5

    # Blend original luminance with radial gradient
    depth = blurred * radial_gradient
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255

    # Apply colormap (turbo-like: blue -> green -> yellow -> red)
    depth_normalized = depth / 255.0
    r = np.clip(1.5 - np.abs(depth_normalized * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(depth_normalized * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(depth_normalized * 4 - 1), 0, 1)

    colormap = np.stack([r * 255, g * 255, b * 255], axis=-1).astype(np.uint8)
    depth_img = Image.fromarray(colormap, "RGB")

    buf = io.BytesIO()
    depth_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@router.post("")
async def run_depth_estimation(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    await asyncio.sleep(2.0)

    depth_b64 = generate_depth_map(image)

    return {
        "status": "success",
        "model": "DPT-Large",
        "min_depth_m": round(random.uniform(0.1, 0.5), 2),
        "max_depth_m": round(random.uniform(5.0, 15.0), 2),
        "image_b64": depth_b64,
        "processing_time_ms": random.randint(1200, 2500),
    }
