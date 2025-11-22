import os
import shutil
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# ============================================================
# 1. CONFIGURATION
# ============================================================
# ตรวจสอบ Path ให้ถูกต้อง
ORIGINAL_PATH = r"C:\Users\rutsa\PycharmProjects\MLproject\MLproject\dataset"
NEW_PATH = r"C:\Users\rutsa\PycharmProjects\MLproject\MLproject\dataset_training_copy"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20  # เพิ่มรอบการเรียนรู้เล็กน้อยเพื่อความแม่นยำ


# ============================================================
# 2. DATASET PREPARATION
# ============================================================
def build_dataset(files, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))

    def load_img(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

        # --- CRITICAL CONFIG ---
        # EfficientNetB0 ต้องการค่าสี 0-255 (float32)
        # ห้ามหาร 255.0 เด็ดขาดสำหรับโมเดลนี้
        img = tf.cast(img, tf.float32)
        return img, label

    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# 3. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("🚀 Starting Training Process (Binary: Real vs AI)...")

    # Setup GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Copy Dataset Safety
    if not os.path.exists(ORIGINAL_PATH):
        print(f"❌ Error: ไม่พบโฟลเดอร์ {ORIGINAL_PATH}")
        exit()

    if not os.path.exists(NEW_PATH):
        print("📂 Creating dataset backup...")
        shutil.copytree(ORIGINAL_PATH, NEW_PATH)
    else:
        print("📂 Using existing dataset backup.")

    dataset_path = NEW_PATH

    # Load Classes (Expected: ['ai', 'real'])
    try:
        classes = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    except:
        print("❌ Error reading dataset")
        exit()

    print(f"✅ Classes Found: {classes}")
    if len(classes) != 2:
        print("⚠️ Warning: แนะนำให้มีแค่ 2 โฟลเดอร์ ('ai', 'real') เพื่อความแม่นยำสูงสุด")

    # Load Image Paths
    file_paths = []
    labels = []

    for label_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(dataset_path, cls_name)
        images = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        file_paths.extend(images)
        labels.extend([label_idx] * len(images))

    file_paths = np.array(file_paths)
    labels = np.array(labels)
    print(f"📸 Total images: {len(file_paths)}")

    if len(file_paths) == 0:
        print("❌ No images found!")
        exit()

    # Split Data (70% Train, 15% Val, 15% Test)
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        file_paths, labels, test_size=0.30, stratify=labels, random_state=42
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
    )

    print(f"📊 Split Stats: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # Build TF Datasets
    train_ds = build_dataset(train_files, train_labels, shuffle=True)
    val_ds = build_dataset(val_files, val_labels)
    test_ds = build_dataset(test_files, test_labels)

    # Data Augmentation (เพิ่มความหลากหลายให้ภาพ)
    data_aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    # Model Architecture (EfficientNetB0)
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze pretrained weights

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_aug(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Output Layer (1 Node for Binary Classification: 0=AI, 1=Real)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # Training
    print("🏋️ Starting Training...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # Evaluation
    print("📝 Evaluating Model...")
    loss, acc = model.evaluate(test_ds)
    print(f"🏆 Test Accuracy: {acc * 100:.2f}%")

    # Save Model
    save_path = 'dog_model_binary.keras'
    print(f"💾 Saving model to {save_path}...")
    model.save(save_path)
    print("✅ Model saved successfully!")

    # Confusion Matrix
    y_true = []
    y_pred = []
    print("📊 Generating Confusion Matrix...")

    for images, lbls in test_ds:
        preds = model.predict(images, verbose=0)
        preds = (preds > 0.5).astype(int).flatten()
        y_true.extend(lbls.numpy())
        y_pred.extend(preds)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    plt.figure(figsize=(6, 6))
    disp.plot(cmap="Blues", ax=plt.gca())
    plt.title(f"Confusion Matrix (Acc: {acc * 100:.2f}%)")
    plt.show()