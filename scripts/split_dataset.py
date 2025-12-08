import os
import shutil
import random

SRC_REAL = "datasets/ffpp/real"
SRC_FAKE = "datasets/ffpp/fake"

DST = "datasets/ffpp"

# Create folders if not exist
for split in ["train", "val"]:
    for cls in ["real", "fake"]:
        os.makedirs(os.path.join(DST, split, cls), exist_ok=True)

# ========== REAL ==========
real_files = [f for f in os.listdir(SRC_REAL) if os.path.isfile(os.path.join(SRC_REAL, f))]
random.shuffle(real_files)
cut = int(0.8 * len(real_files))

train_real = real_files[:cut]
val_real = real_files[cut:]

for f in train_real:
    shutil.copy2(os.path.join(SRC_REAL, f), os.path.join(DST, "train", "real"))

for f in val_real:
    shutil.copy2(os.path.join(SRC_REAL, f), os.path.join(DST, "val", "real"))

# ========== FAKE ==========
fake_files = [f for f in os.listdir(SRC_FAKE) if os.path.isfile(os.path.join(SRC_FAKE, f))]
random.shuffle(fake_files)
cut = int(0.8 * len(fake_files))

train_fake = fake_files[:cut]
val_fake = fake_files[cut:]

for f in train_fake:
    shutil.copy2(os.path.join(SRC_FAKE, f), os.path.join(DST, "train", "fake"))

for f in val_fake:
    shutil.copy2(os.path.join(SRC_FAKE, f), os.path.join(DST, "val", "fake"))

print("DONE — 80/20 split created inside datasets/ffpp")
