file_path = "/home/lvu5/LibMultiLabel/data/LF-AmazonTitles-131K/zeroshot/trn.txt"

prev_idx = None

with open(file_path, "r") as f:
    for line_num, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('\t')
        if len(parts) < 1:
            continue
        
        idx = int(parts[0])
        
        if prev_idx is not None and idx != prev_idx + 1:
            print(f"Issue at line {line_num}: expected {prev_idx + 1}, got {idx}")
        
        prev_idx = idx

print("Done checking.")