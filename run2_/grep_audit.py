targets = [
    '135.0', '69.9', '73.7', '49.1', '10.4', '16.4', '7.6', '65.6', '90.2',
    '44.1', '57.8', '21.0', '15.5', '15.3', '11.1', '12.4', '80.74', '135.67',
    '1920', '200 timed', '50 warm-up'
]
files = ['paper/main.tex', 'README.md']

print("=====================================================================")
print("                   GREP AUDIT FOR TARGET STRINGS                    ")
print("=====================================================================")

for fpath in files:
    print(f"\n--- Scanning {fpath} ---")
    with open(fpath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    hits = 0
    for idx, line in enumerate(lines, 1):
        for t in targets:
            if t in line:
                hits += 1
                print(f"Target: '{t}' | Line {idx}:")
                print(f"  {line.strip()[:130]}")
    if hits == 0:
        print("  --> No matches found.")
    else:
        print(f"  --> Total hits in {fpath}: {hits}")

print("\nAudit completed.")
