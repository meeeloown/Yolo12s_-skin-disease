import os
import yaml
import random
from collections import defaultdict

# === 1. SETUP PATH ===
base_path = r"C:\BERKAS KULIAH\Yolov\Final mekaMedis\Dataset_baru\skin disease 2.v2i.yolov12_manualSplit\finalDataset"
data_yaml_path = os.path.join(base_path, "data.yaml")

# === 2. BACA NAMA CLASS DARI data.yaml ===
with open(data_yaml_path, 'r') as f:
    data_yaml = yaml.safe_load(f)
    class_names = data_yaml['names']
    num_classes = data_yaml['nc']

# === 3. INISIALISASI ===
splits = ['train', 'val', 'test']
image_extensions = ['.jpg', '.jpeg', '.png']

# Struktur hitung jumlah gambar per kelas per split
class_split_counts = {class_id: {split: 0 for split in splits} for class_id in range(num_classes)}

# Untuk cek gambar tanpa label dan duplikat
non_labeled_images = {split: [] for split in splits}
seen_images = {split: [] for split in splits}  # pake list supaya bisa count duplikat

# Kumpulan semua gambar per split (gabungan semua split)
all_images_combined = []

# === 4. PROSES SETIAP SPLIT ===
for split in splits:
    image_dir = os.path.join(base_path, 'images', split)
    label_dir = os.path.join(base_path, 'labels', split)

    all_images = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in image_extensions]
    all_images_combined.extend([f"{split}/{img}" for img in all_images])  # simpan nama dengan prefix split

    for image_file in all_images:
        seen_images[split].append(image_file)
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(label_dir, label_file)

        if os.path.exists(label_path):
            class_ids_in_image = set()
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_ids_in_image.add(class_id)
            for class_id in class_ids_in_image:
                class_split_counts[class_id][split] += 1
        else:
            non_labeled_images[split].append(image_file)

# === 5. HITUNG DUPLIKAT GAMBAR PER SPLIT ===
duplicate_counts = {}
for split in splits:
    counts = defaultdict(int)
    for img in seen_images[split]:
        counts[img] += 1
    duplicates = [img for img, cnt in counts.items() if cnt > 1]
    duplicate_counts[split] = len(set(duplicates))

# === 6. TAMPILKAN HASIL ANALISIS ===
print("=== ANALISIS DATASET ===\n")

print("Jumlah gambar per kelas di setiap split:")
for class_id, split_counts in class_split_counts.items():
    class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
    counts = [f"{split.capitalize()}={split_counts[split]}" for split in splits]
    print(f"{class_name} (ID {class_id}): " + ", ".join(counts))

print("\n=== GAMBAR TANPA LABEL ===")
for split in splits:
    no_label = non_labeled_images[split]
    print(f"{split.capitalize()} - {len(no_label)} gambar tanpa label.")

print("\n=== DUPLIKAT NAMA FILE ===")
for split in splits:
    print(f"{split.capitalize()} - {duplicate_counts[split]} gambar duplikat.")

# === 7. HITUNG TOTAL GAMBAR (unik) PER SPLIT ===
unique_images_per_split = {split: len(set(seen_images[split])) for split in splits}
total_unique_images = sum(unique_images_per_split.values())

print("\nJumlah gambar unik per split:")
for split in splits:
    print(f"{split.capitalize()}: {unique_images_per_split[split]} gambar")
print(f"Total gambar unik di semua split: {total_unique_images}")