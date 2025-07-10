import csv
import os

RAW_DIR = os.path.join('data', 'raw')

files = [
    ('train.csv', 'train_small.csv', 2000),
    ('val.csv', 'val_small.csv', 200),
    ('test.csv', 'test_small.csv', 200),
]

def make_small_csv(infile, outfile, n):
    in_path = os.path.join(RAW_DIR, infile)
    out_path = os.path.join(RAW_DIR, outfile)
    with open(in_path, encoding='utf-8') as fin, open(out_path, 'w', encoding='utf-8', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for i, row in enumerate(reader):
            if i >= n:
                break
            writer.writerow(row)
if __name__ == '__main__':
    for infile, outfile, n in files:
        make_small_csv(infile, outfile, n)
    print('Small CSVs generated.') 