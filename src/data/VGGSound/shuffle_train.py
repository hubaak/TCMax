import csv
import random

input_csv = "train_origin.csv"
output_csv = "train.csv"

with open(input_csv, encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    headers = next(reader)
    data_rows = list(reader)

random.seed(42)
random.shuffle(data_rows)

with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(data_rows)

print(f"Shuffled csv is saved in {output_csv}")