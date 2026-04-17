import os
import random
import shutil
from collections import defaultdict

# Direktori dataset awal
base_dir = r"C:\BERKAS KULIAH\Yolov\Final mekaMedis\Dataset_baru\skin disease 2.v2i.yolov12_manualSplit\train"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

# Direktori output split (1 level di atas 'train')
output_base = os.path.join(base_dir, "..")
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_base, f"images/{split}"), exist_ok=True)
    os.makedirs(os.path.join(output_base, f"labels/{split}"), exist_ok=True)

# Kumpulkan gambar per kelas
class_to_images = defaultdict(list)

for img_file in os.listdir(images_dir):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(labels_dir, label_file)
    
    if not os.path.exists(label_path):
        continue  # skip gambar tanpa label
    
    # Baca kelas dari label (anggap hanya 1 kelas per file)
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        if not lines:
            continue
        class_id = int(lines[0].split()[0])  # ambil kelas dari baris pertama
        class_to_images[class_id].append(img_file)

# Proporsi split
train_pct = 0.8
val_pct = 0.1
test_pct = 0.1

# Split per kelas
final_splits = {
    "train": set(),
    "val": set(),
    "test": set()
}

random.seed(42)  # supaya hasil split sama tiap kali run

for class_id, imgs in class_to_images.items():
    imgs = list(set(imgs))  # pastikan unik
    random.shuffle(imgs)
    
    n = len(imgs)
    n_train = int(train_pct * n)
    n_val = int(val_pct * n)
    n_test = n - n_train - n_val
    
    final_splits["train"].update(imgs[:n_train])
    final_splits["val"].update(imgs[n_train:n_train + n_val])
    final_splits["test"].update(imgs[n_train + n_val:])

# Salin file ke folder split
for split, imgs in final_splits.items():
    print(f"Split {split}: {len(imgs)} images")
    for img_file in imgs:
        label_file = os.path.splitext(img_file)[0] + ".txt"
        shutil.copyfile(os.path.join(images_dir, img_file), os.path.join(output_base, f"images/{split}", img_file))
        shutil.copyfile(os.path.join(labels_dir, label_file), os.path.join(output_base, f"labels/{split}", label_file))

print("Split selesai.")
