import asyncio
import base64
import io
import random
import math
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import numpy as np

router = APIRouter()


def generate_point_cloud(image: Image.Image, depth_scale: float = 1.0):
    """Generate a placeholder 3D point cloud from image."""
    img_small = image.resize((64, 64)).convert("RGB")
    img_array = np.array(img_small)

    # Simulate depth values
    gray = np.mean(img_array, axis=2) / 255.0
    from PIL import ImageFilter
    gray_img = Image.fromarray((gray * 255).astype(np.uint8))
    blurred = np.array(gray_img.filter(ImageFilter.GaussianBlur(radius=3))) / 255.0

    points = []
    colors = []

    h, w = blurred.shape
    cx, cy = w / 2, h / 2

    for y in range(h):
        for x in range(w):
            depth = blurred[y, x] * depth_scale * 3.0
            # Back-project to 3D
            px = (x - cx) / w * 2.0
            py = -(y - cy) / h * 2.0
            pz = depth

            r, g, b = img_array[y, x]
            points.append([round(px, 4), round(py, 4), round(pz, 4)])
            colors.append([round(r / 255, 3), round(g / 255, 3), round(b / 255, 3)])

    return points, colors


def generate_mesh_faces(w: int, h: int):
    """Generate mesh connectivity for a grid of w*h points."""
    faces = []
    for y in range(h - 1):
        for x in range(w - 1):
            i = y * w + x
            faces.append([i, i + 1, i + w])
            faces.append([i + 1, i + w + 1, i + w])
    return faces


@router.post("")
async def run_reconstruction(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    await asyncio.sleep(3.0)

    points, colors = generate_point_cloud(image)
    faces = generate_mesh_faces(64, 64)

    # Sample to reduce payload size
    sample_rate = 4
    sampled_points = points[::sample_rate]
    sampled_colors = colors[::sample_rate]

    return {
        "status": "success",
        "model": "NeRF-Lite",
        "num_points": len(sampled_points),
        "bounding_box": {
            "x": [-1.0, 1.0],
            "y": [-1.0, 1.0],
            "z": [0.0, 3.0],
        },
        "point_cloud": {
            "positions": sampled_points,
            "colors": sampled_colors,
        },
        "processing_time_ms": random.randint(2500, 4500),
    }
