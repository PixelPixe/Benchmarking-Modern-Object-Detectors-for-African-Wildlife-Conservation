import os
import csv
import hashlib
from pathlib import Path
from PIL import Image
import numpy as np

# Load 63 duplicate pairs from results_v2/duplicates_audit.csv
pairs = []
with open('results_v2/duplicates_audit.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        pairs.append(row)

print(f"Loaded {len(pairs)} pairs.")

def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_pixel_diff(path1, path2, target_size=(512, 512)):
    im1 = Image.open(path1).convert('RGB').resize(target_size)
    im2 = Image.open(path2).convert('RGB').resize(target_size)
    arr1 = np.array(im1, dtype=np.float32)
    arr2 = np.array(im2, dtype=np.float32)
    mad = np.mean(np.abs(arr1 - arr2))
    return mad

results = []
byte_identical_count = 0

for p in pairs:
    f1 = Path(p['Dropped_File'])
    f2 = Path(p['Kept_File'])
    
    sha1 = get_sha256(f1)
    sha2 = get_sha256(f2)
    is_byte_identical = (sha1 == sha2)
    if is_byte_identical:
        byte_identical_count += 1
        
    mad = get_pixel_diff(f1, f2)
    
    results.append({
        'index': p['Index'],
        'dropped': p['Dropped_File'],
        'kept': p['Kept_File'],
        'byte_identical': is_byte_identical,
        'sha256_match': 'Yes' if is_byte_identical else 'No',
        'mean_abs_pixel_diff': round(float(mad), 4)
    })

print(f"\n--- DUPLICATE PAIRS BYTE & PIXEL ANALYSIS ---")
print(f"Total duplicate pairs: {len(pairs)}")
print(f"Byte-identical pairs (SHA-256 match): {byte_identical_count} / {len(pairs)} ({byte_identical_count/len(pairs)*100:.1f}%)")
non_identical = len(pairs) - byte_identical_count
print(f"Non-byte-identical pairs (different compression / metadata): {non_identical} / {len(pairs)} ({non_identical/len(pairs)*100:.1f}%)")

mads = [r['mean_abs_pixel_diff'] for r in results]
print(f"Mean Absolute Pixel Difference (MAD) across all 63 pairs:")
print(f"  Min MAD: {min(mads):.4f}")
print(f"  Max MAD: {max(mads):.4f}")
print(f"  Mean MAD: {np.mean(mads):.4f}")
print(f"  Median MAD: {np.median(mads):.4f}")

# Save detailed results
with open('results_v2/duplicate_pairs_pixel_analysis.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print("Saved detailed results to results_v2/duplicate_pairs_pixel_analysis.csv")

# Now inspect the two test/train pairs at distance 8 and 10
# Pair 1: test/1 (41).jpg <-> train/1 (81).jpg (distance 8)
# Pair 2: test/4 (360).jpg <-> train/1 (270).jpg (distance 10)
print("\n--- NEAR-DUPLICATE PAIRS INSPECTION ---")
p1_test = Path('dataset/images/test/1 (41).jpg')
p1_train = Path('dataset/images/train/1 (81).jpg')
mad_p1 = get_pixel_diff(p1_test, p1_train)

p2_test = Path('dataset/images/test/4 (360).jpg')
p2_train = Path('dataset/images/train/1 (270).jpg')
mad_p2 = get_pixel_diff(p2_test, p2_train)

print(f"Pair 1 (d=8): {p1_test} vs {p1_train} | MAD: {mad_p1:.4f}")
print(f"Pair 2 (d=10): {p2_test} vs {p2_train} | MAD: {mad_p2:.4f}")

# Create side-by-side composite images
def make_side_by_side(p1, p2, out_name, title1, title2):
    im1 = Image.open(p1).convert('RGB')
    im2 = Image.open(p2).convert('RGB')
    
    # Target height 600
    h = 600
    w1 = int(im1.width * (h / im1.height))
    w2 = int(im2.width * (h / im2.height))
    
    im1_r = im1.resize((w1, h))
    im2_r = im2.resize((w2, h))
    
    composite = Image.new('RGB', (w1 + w2 + 20, h + 60), color=(255, 255, 255))
    composite.paste(im1_r, (0, 50))
    composite.paste(im2_r, (w1 + 20, 50))
    
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(composite)
    # Default font
    draw.text((10, 15), title1, fill=(0, 0, 0))
    draw.text((w1 + 30, 15), title2, fill=(0, 0, 0))
    
    os.makedirs('results_v2/inspection_images', exist_ok=True)
    out_path = Path('results_v2/inspection_images') / out_name
    composite.save(out_path)
    print(f"Saved composite image to {out_path}")

make_side_by_side(p1_test, p1_train, 'pair1_dist8_buffalo.jpg', 
                  'TEST: 1 (41).jpg (Buffalo)', 'TRAIN: 1 (81).jpg (Buffalo)')
make_side_by_side(p2_test, p2_train, 'pair2_dist10_zebra_buffalo.jpg', 
                  'TEST: 4 (360).jpg (Zebra)', 'TRAIN: 1 (270).jpg (Buffalo)')
