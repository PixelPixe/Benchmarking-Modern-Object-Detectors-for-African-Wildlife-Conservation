import os, json, random, csv
from pathlib import Path
import imagehash
from PIL import Image

dataset_dir = Path('datasets/african-wildlife')
all_files = list(dataset_dir.rglob('*'))
image_paths = [p for p in all_files if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]

def get_image_hash(filepath):
    try:
        return str(imagehash.phash(Image.open(filepath)))
    except:
        return None

hash_to_first = {}
duplicates = []

for p in image_paths:
    h = get_image_hash(p)
    if not h:
        continue
    if h in hash_to_first:
        first_p = hash_to_first[h]
        duplicates.append((p, first_p, h))
    else:
        hash_to_first[h] = p

out_rows = []
partition_counts = {}
for i, (dropped, kept, h) in enumerate(duplicates, 1):
    d_part = dropped.parent.name
    k_part = kept.parent.name
    spans = 'Yes' if d_part != k_part else 'No'
    pair_type = f"{k_part} -> {d_part}"
    partition_counts[pair_type] = partition_counts.get(pair_type, 0) + 1
    out_rows.append({
        'Index': i,
        'Dropped_File': str(dropped).replace('\\', '/'),
        'Dropped_Partition': d_part,
        'Kept_File': str(kept).replace('\\', '/'),
        'Kept_Partition': k_part,
        'Spans_Partitions': spans,
        'pHash': h
    })

os.makedirs('results_v2', exist_ok=True)
with open('results_v2/duplicates_audit.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    writer.writeheader()
    writer.writerows(out_rows)

print(f"Total duplicates found: {len(duplicates)}")
print('Partition breakdown of duplicate pairs (Kept -> Dropped):')
for k, v in sorted(partition_counts.items()):
    print(f'  {k}: {v}')

spanning = [r for r in out_rows if r['Spans_Partitions'] == 'Yes']
print(f"Total spanning pairs: {len(spanning)} / {len(out_rows)} ({len(spanning)/len(out_rows)*100:.1f}%)")

random.seed(42)
sample_5 = random.sample(out_rows, 5)
print('\n5 Random Pairs:')
for s in sample_5:
    print(f"  [Index {s['Index']}] Dropped: {s['Dropped_File']} (Split: {s['Dropped_Partition']}) | Kept: {s['Kept_File']} (Split: {s['Kept_Partition']}) | Spans Partitions: {s['Spans_Partitions']}")
