import os
from pathlib import Path
from PIL import Image
import imagehash
from tqdm import tqdm

train_dir = Path('dataset/images/train')
test_dir = Path('dataset/images/test')

train_images = list(train_dir.glob('*.*'))
test_images = list(test_dir.glob('*.*'))

print(f"Loaded {len(train_images)} train images and {len(test_images)} test images.")

# Compute pHash for all train images
train_hashes = []
for p in tqdm(train_images, desc="Hashing train images"):
    try:
        h = imagehash.phash(Image.open(p))
        train_hashes.append((p.name, h))
    except Exception as e:
        print(f"Error hashing train image {p}: {e}")

# Compute pHash for all test images
test_hashes = []
for p in tqdm(test_images, desc="Hashing test images"):
    try:
        h = imagehash.phash(Image.open(p))
        test_hashes.append((p.name, h))
    except Exception as e:
        print(f"Error hashing test image {p}: {e}")

min_dists = []
details = []

for t_name, t_h in tqdm(test_hashes, desc="Comparing test to train"):
    min_d = 999
    best_train = None
    for tr_name, tr_h in train_hashes:
        d = t_h - tr_h  # Hamming distance in imagehash
        if d < min_d:
            min_d = d
            best_train = tr_name
            if min_d == 0:
                break
    min_dists.append(min_d)
    details.append((t_name, best_train, min_d))

# Thresholds: <= 0, <= 4, <= 8, <= 10
c_0 = sum(1 for d in min_dists if d == 0)
c_4 = sum(1 for d in min_dists if d <= 4)
c_8 = sum(1 for d in min_dists if d <= 8)
c_10 = sum(1 for d in min_dists if d <= 10)

print("\n--- RESULTS ---")
print(f"Total test images: {len(test_images)}")
print(f"Test images with train image within Hamming distance == 0: {c_0} ({c_0/len(test_images)*100:.2f}%)")
print(f"Test images with train image within Hamming distance <= 4: {c_4} ({c_4/len(test_images)*100:.2f}%)")
print(f"Test images with train image within Hamming distance <= 8: {c_8} ({c_8/len(test_images)*100:.2f}%)")
print(f"Test images with train image within Hamming distance <= 10: {c_10} ({c_10/len(test_images)*100:.2f}%)")

print("\nDistribution of minimum Hamming distance from test to nearest train image:")
from collections import Counter
dist_counts = Counter(min_dists)
for d in sorted(dist_counts.keys()):
    print(f"  Distance {d:2d}: {dist_counts[d]:3d} test images")

# Save results
import csv
os.makedirs('results_v2', exist_ok=True)
with open('results_v2/test_to_train_hamming_distances.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['test_image', 'nearest_train_image', 'min_hamming_distance'])
    for row in details:
        writer.writerow(row)
print("Saved to results_v2/test_to_train_hamming_distances.csv")
