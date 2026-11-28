"""
Dataset Downloader for PlantVillage Dataset
Downloads real leaf images from the official PlantVillage repository on GitHub.
"""

import os
import sys
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
import io

# Classes to download (mapped to repo folder names)
DEFAULT_CLASSES = [
    ("Pepper__bell___Bacterial_spot", "Pepper,_bell___Bacterial_spot"),
    ("Pepper__bell___healthy", "Pepper,_bell___healthy"),
    ("Potato___Early_blight", "Potato___Early_blight"),
    ("Potato___Late_blight", "Potato___Late_blight"),
    ("Potato___healthy", "Potato___healthy"),
    ("Tomato_Bacterial_spot", "Tomato___Bacterial_spot"),
    ("Tomato_Early_blight", "Tomato___Early_blight"),
    ("Tomato_Late_blight", "Tomato___Late_blight"),
    ("Tomato_Leaf_Mold", "Tomato___Leaf_Mold"),
    ("Tomato_Septoria_leaf_spot", "Tomato___Septoria_leaf_spot"),
    ("Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato___Spider_mites Two-spotted_spider_mite"),
    ("Tomato__Target_Spot", "Tomato___Target_Spot"),
    ("Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"),
    ("Tomato__Tomato_mosaic_virus", "Tomato___Tomato_mosaic_virus"),
    ("Tomato_healthy", "Tomato___healthy"),
]

GITHUB_API_BASE = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color"


def get_image_list_for_class(repo_folder: str, limit: int = 25):
    """Fetch list of image download URLs for a given class folder."""
    import urllib.parse
    encoded_folder = urllib.parse.quote(repo_folder)
    url = f"{GITHUB_API_BASE}/{encoded_folder}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            items = json.loads(resp.read().decode())
            image_items = [item for item in items if item["name"].lower().endswith((".jpg", ".jpeg", ".png"))]
            return image_items[:limit]
    except Exception as e:
        print(f"  [WARN] Failed to fetch folder list for {repo_folder}: {e}")
        return []


def download_single_image(download_url: str, save_path: Path):
    """Download and validate a single image."""
    if save_path.exists() and save_path.stat().st_size > 1000:
        return True
    try:
        req = urllib.request.Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            # Validate with PIL
            img = Image.open(io.BytesIO(data))
            img.verify()
            with open(save_path, "wb") as f:
                f.write(data)
            return True
    except Exception as e:
        # print(f"  Failed {save_path.name}: {e}")
        return False


def download_plantvillage_dataset(output_dir: str = "data/plantvillage", images_per_class: int = 25):
    """Download a real PlantVillage dataset subset for training."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🌿 Downloading genuine PlantVillage dataset to: {output_path.resolve()}")
    print(f"📊 Target: {len(DEFAULT_CLASSES)} classes, {images_per_class} images per class")
    
    total_downloaded = 0
    
    for local_name, repo_name in DEFAULT_CLASSES:
        class_dir = output_path / local_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Check existing images
        existing = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.JPG")) + list(class_dir.glob("*.png"))
        if len(existing) >= images_per_class:
            print(f"  ✓ {local_name}: already has {len(existing)} images")
            total_downloaded += len(existing)
            continue
            
        print(f"  Fetching image list for {local_name}...")
        image_items = get_image_list_for_class(repo_name, limit=images_per_class)
        if not image_items:
            print(f"  ✗ No images found for {repo_name}")
            continue
            
        # Download in parallel
        download_tasks = []
        with ThreadPoolExecutor(max_workers=6) as executor:
            for item in image_items:
                dest = class_dir / item["name"]
                download_tasks.append(executor.submit(download_single_image, item["download_url"], dest))
                
            success_count = sum(1 for t in as_completed(download_tasks) if t.result())
            
        print(f"  ✓ {local_name}: downloaded {success_count} images")
        total_downloaded += success_count
        
    print(f"\n🎉 Dataset ready! Total images: {total_downloaded}")
    return str(output_path.resolve())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download real PlantVillage leaf dataset")
    parser.add_argument("--output-dir", default="data/plantvillage", help="Directory to save images")
    parser.add_argument("--per-class", type=int, default=25, help="Number of images per class")
    args = parser.parse_args()
    
    download_plantvillage_dataset(args.output_dir, args.per_class)
