import os
import time
import random
import math
import re
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from ultralytics import YOLO
import easyocr
import pytesseract

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PaddleOCR = None
    PADDLE_AVAILABLE = False
    print("[INFO] PaddleOCR nie jest zainstalowany. "
          "Tryb Paddle będzie pominięty (zostanie EasyOCR).")


# ============================================
# KONFIGURACJA
# ============================================

USE_KAGGLEHUB = True  # dla ręcznego pobrania datasetu - False

LOCAL_DATASET_ROOT = Path("data/poland-vehicle-license-plate-dataset")
YOLO_ROOT = Path("yolo_lp")

TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

TRAIN_YOLO = False
DEBUG_SHOW_CROPS = False
EVAL_MAX_IMAGES = 100
USE_PADDLE = False


# ============================================
# ŚRODOWISKO (device, Tesseract)
# ============================================

plt.rcParams["figure.figsize"] = (8, 5)

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
DEVICE = 0 if torch.cuda.is_available() else "cpu"

if os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
else:
    print(
        "[UWAGA] Ścieżka do Tesseracta nie istnieje. "
        "Zainstaluj Tesseract i popraw TESSERACT_CMD na górze pliku."
    )


# ============================================
# 1. Wczytanie / pobranie datasetu
# ============================================

def get_dataset_root() -> Path:
    if USE_KAGGLEHUB:
        import kagglehub

        print("Pobieram dataset z Kaggle przy pomocy kagglehub...")
        dataset_path = kagglehub.dataset_download(
            "piotrstefaskiue/poland-vehicle-license-plate-dataset"
        )
        print("Dataset downloaded")
        return Path(dataset_path)
    else:
        print("Używam lokalnego datasetu:", LOCAL_DATASET_ROOT)
        if not LOCAL_DATASET_ROOT.exists():
            raise FileNotFoundError(
                f"Nie znaleziono katalogu z datasetem: {LOCAL_DATASET_ROOT}\n"
                f"Utwórz go i rozpakuj tam pliki (annotations.xml + photos/)."
            )
        return LOCAL_DATASET_ROOT


def load_annotations(dataset_root: Path):
    ann_path = dataset_root / "annotations.xml"
    photos_dir = dataset_root / "photos"

    print("annotations.xml:", ann_path.exists(), ann_path)
    print("photos dir:", photos_dir.exists(), photos_dir)

    if not ann_path.exists() or not photos_dir.exists():
        raise FileNotFoundError(
            "Brak annotations.xml albo katalogu photos w dataset_root."
        )

    tree = ET.parse(ann_path)
    root_xml = tree.getroot()

    records = []
    for image in root_xml.findall("image"):
        name = image.attrib["name"]
        w = int(image.attrib["width"])
        h = int(image.attrib["height"])
        img_path = photos_dir / name

        for box in image.findall("box"):
            xtl = float(box.attrib["xtl"])
            ytl = float(box.attrib["ytl"])
            xbr = float(box.attrib["xbr"])
            ybr = float(box.attrib["ybr"])

            plate_text = None
            for attr in box.findall("attribute"):
                if attr.text and attr.text.strip():
                    plate_text = attr.text.strip()
                    break

            records.append(
                {
                    "image_name": name,
                    "image_path": str(img_path),
                    "img_w": w,
                    "img_h": h,
                    "bbox": (xtl, ytl, xbr, ybr),
                    "plate_text": plate_text,
                }
            )

    print("Łącznie rekordów:", len(records))

    usable_records = [
        r
        for r in records
        if r["plate_text"] is not None and os.path.exists(r["image_path"])
    ]
    print("Rekordy z tekstem i istniejącym plikiem:", len(usable_records))
    if usable_records:
        print("Przykład:", usable_records[0])

    return usable_records


# ============================================
# 2. Podział na train / val + podgląd
# ============================================

def split_train_val(usable_records, train_ratio=0.7):
    random.seed(42)
    random.shuffle(usable_records)
    split_idx = int(train_ratio * len(usable_records))
    train_records = usable_records[:split_idx]
    val_records = usable_records[split_idx:]
    print(f"Train: {len(train_records)}")
    print(
        f"Val:   {len(val_records)} ({100*len(val_records)/len(usable_records):.1f}%)"
    )
    return train_records, val_records


def show_samples(records_list, n=6):
    samples = random.sample(records_list, min(n, len(records_list)))
    cols = 3
    rows = math.ceil(len(samples) / cols)
    plt.figure(figsize=(cols * 4, rows * 3))

    for i, rec in enumerate(samples, 1):
        img = cv2.imread(rec["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = map(int, rec["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.title(rec["plate_text"])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# ============================================
# 3. Konwersja do formatu YOLO
# ============================================

def prepare_yolo_dirs(root: Path):
    images_train = root / "images" / "train"
    images_val = root / "images" / "val"
    labels_train = root / "labels" / "train"
    labels_val = root / "labels" / "val"

    for p in [images_train, images_val, labels_train, labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    return images_train, images_val, labels_train, labels_val


def write_yolo_label(label_path: Path, img_w: int, img_h: int, boxes):
    lines = []
    for (x1, y1, x2, y2) in boxes:
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def export_split(rec_list, img_dst_dir, lbl_dst_dir):
    count = 0
    grouped = {}
    for r in rec_list:
        grouped.setdefault(r["image_name"], []).append(r)

    for name, rs in grouped.items():
        src_path = Path(rs[0]["image_path"])
        if not src_path.exists():
            continue
        dst_img = img_dst_dir / name
        shutil.copy2(src_path, dst_img)

        w = rs[0]["img_w"]
        h = rs[0]["img_h"]
        boxes = [r["bbox"] for r in rs]
        dst_lbl = lbl_dst_dir / (Path(name).stem + ".txt")
        write_yolo_label(dst_lbl, w, h, boxes)
        count += 1
    return count


def create_yolo_dataset(train_records, val_records, root: Path):
    images_train, images_val, labels_train, labels_val = prepare_yolo_dirs(root)

    n_train = export_split(train_records, images_train, labels_train)
    n_val = export_split(val_records, images_val, labels_val)

    print("YOLO train images:", n_train)
    print("YOLO val images:", n_val)

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        f"""
path: {root}
train: images/train
val: images/val
names:
  0: plate
""".strip(),
        encoding="utf-8",
    )
    print("Zapisano data.yaml:")
    print(data_yaml.read_text())
    return data_yaml


# ============================================
# 4. Trening YOLOv8n
# ============================================

def train_yolo_model(data_yaml: Path, root: Path) -> YOLO:
    model = YOLO("yolov8n.pt")

    print("Rozpoczynam trening YOLO...")
    _ = model.train(
        data=str(data_yaml),
        imgsz=640,
        epochs=30,
        batch=16,
        device=DEVICE,
        workers=2,
        verbose=False,
        project=str(root / "runs"),
        name="yolov8n_lp",
        exist_ok=True,
    )

    print("Trening zakończony.")
    print("Najlepszy model:", model.trainer.best)
    best_model = YOLO(model.trainer.best)
    return best_model


def load_best_model(root: Path) -> YOLO:
    """
    Szuka pliku best.pt w katalogu 'runs' (domyślne wyjście Ultralytics).
    """
    runs_dir = Path("runs")

    if not runs_dir.exists():
        raise FileNotFoundError("Brak katalogu 'runs' w katalogu projektu.")

    candidates = list(runs_dir.rglob("best.pt"))
    if not candidates:
        raise FileNotFoundError("Nie znaleziono żadnego best.pt w katalogu 'runs'.")

    best_path = candidates[0]
    print("Ładuję istniejący model:", best_path)
    return YOLO(str(best_path))


# ============================================
# 5. IoU + powiększanie bbox
# ============================================

def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0
    return inter_area / union


def expand_bbox(bbox, img_shape, scale=0.15):
    x1, y1, x2, y2 = bbox
    h, w = img_shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    dx = int(bw * scale)
    dy = int(bh * scale)

    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(w, x2 + dx)
    ny2 = min(h, y2 + dy)

    return nx1, ny1, nx2, ny2


def trim_left_pl_band(crop, pct=0.08):
    """
    Odcina pasek po lewej stronie cropa (np. 'PL'),
    aby OCR nie łapał zbędnych znaków na początku.
    """
    if crop is None or crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    cut = int(w * pct)
    if 0 < cut < w:
        return crop[:, cut:]
    return crop


# ============================================
# 6. OCR – helpery
# ============================================

ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

paddle_reader = None
if PADDLE_AVAILABLE:
    try:
        paddle_reader = PaddleOCR(
            lang="en",
            use_gpu=False,
            use_angle_cls=False,
            det_db_thesh=0.2,
            det_db_box_thresh=0.3,
            det_db_unclip_ratio=2.0,
        )
        print("PaddleOCR initialized.")
    except Exception as e:
        print(f"[WARN] Nie udało się zainicjalizować PaddleOCR: {e}")
        PADDLE_AVAILABLE = False


def create_variants(img):
    """
    Tworzy warianty preprocessingu (dla prostych testów OCR).
    """
    variants = []

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape
    scale = 80 / h
    new_w = int(w * scale)
    base = cv2.resize(gray, (new_w, 80), interpolation=cv2.INTER_CUBIC)

    padded = cv2.copyMakeBorder(
        base, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255
    )
    variants.append(("base", padded))

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrasted = clahe.apply(base)
    padded = cv2.copyMakeBorder(
        contrasted, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255
    )
    variants.append(("clahe", padded))

    blurred = cv2.GaussianBlur(base, (3, 3), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    padded = cv2.copyMakeBorder(
        binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255
    )
    variants.append(("otsu", padded))

    adaptive = cv2.adaptiveThreshold(
        base,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    padded = cv2.copyMakeBorder(
        adaptive, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255
    )
    variants.append(("adaptive", padded))

    return variants


def clean_text(text):
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def smart_fix(text):
    """
    Pozycyjne poprawki (helper, obecnie nieużywany globalnie).
    """
    if len(text) < 5:
        return text

    fixed = list(text)

    for i in range(min(3, len(text))):
        c = fixed[i]
        if c == "0":
            fixed[i] = "O"
        elif c == "1":
            fixed[i] = "I"
        elif c == "5":
            fixed[i] = "S"
        elif c == "8":
            fixed[i] = "B"
        elif c == "6":
            fixed[i] = "G"

    for i in range(3, min(len(text) - 1, 7)):
        c = fixed[i]
        if c == "O":
            fixed[i] = "0"
        elif c == "I":
            fixed[i] = "1"
        elif c == "Z":
            fixed[i] = "2"
        elif c == "S":
            fixed[i] = "5"
        elif c == "B":
            fixed[i] = "8"
        elif c == "G":
            fixed[i] = "6"

    result = "".join(fixed)

    if len(result) > 8:
        match = re.search(r"^[A-Z]{2,3}[0-9]{4,5}[A-Z0-9]?", result)
        if match:
            result = match.group()
        else:
            result = result[:8]

    return result


def validate_plate(text):
    """
    Sprawdza czy tekst wygląda jak polska tablica + prosty score.
    """
    if not text or len(text) < 5:
        return False, 0

    score = 0

    if len(text) == 7:
        score += 10
    elif len(text) == 8:
        score += 8
    elif len(text) == 6:
        score += 5
    else:
        score -= 5

    if re.match(r"^[A-Z]{2,3}[0-9]{4,5}[A-Z0-9]?$", text):
        score += 15

    if len(text) >= 2 and text[-1] == text[-2]:
        score -= 3

    return (score > 10), score


def plate_accuracy_strict(pred: str, gt: str) -> float:
    """
    - jeśli pred albo gt puste -> 0%
    - jeśli identyczne (po usunięciu spacji) -> 100%
    - inaczej 0%
    """
    if not pred or not gt:
        return 0.0
    return 100.0 if pred.replace(" ", "") == gt.replace(" ", "") else 0.0


def load_gt_simple(dataset_root: Path):
    """
    Ładuje GT:
    - annotations.xml
    - pierwszy <box>/<attribute> jako tablica
    """
    ann_path = dataset_root / "annotations.xml"
    tree = ET.parse(ann_path)
    root = tree.getroot()

    gt = {}
    for image in root.findall("image"):
        name = image.attrib.get("name")
        box = image.find("box")
        if name is None or box is None:
            continue
        attr = box.find("attribute")
        if attr is not None and attr.text:
            gt[name] = attr.text.strip().upper()

    return gt


def yolo_detect_single(img_path: str, model: YOLO):
    """
    Detekcja:
    - YOLO z conf=0.01,
    - bierzemy box o najwyższym confidence,
    - zwracamy crop tablicy i cały obraz.
    """
    image = cv2.imread(img_path)
    if image is None:
        return None, None

    results = model(
        image,
        conf=0.01,
        imgsz=640,
        device=DEVICE,
        verbose=False
    )[0]

    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return None, None

    best_idx = boxes.conf.argmax()
    x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].cpu().numpy())

    plate = image[y1:y2, x1:x2]
    return plate, image


def preprocess_plate_maciek_style(plate_image):
    """
    Preprocessing:
    gray + bilateral blur + binaryzacja.
    """
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)
    return gray, thresh


def evaluate_paddle(best_model: YOLO, dataset_root: Path, max_images=100):
    """
    Ewaluacja soft:
    - losuje do max_images obrazów,
    - brak detekcji -> pomijamy obraz,
    - OCR na cropie (gray, potem thresh),
    - accuracy per obraz: 0% lub 100%,
    - wynik: średnia z tych wartości.
    """
    gt_data = load_gt_simple(dataset_root)

    image_names = list(gt_data.keys())
    random.seed(42)
    random.shuffle(image_names)
    sample = image_names[:min(max_images, len(image_names))]

    total_acc = 0.0
    count = 0

    t0 = time.perf_counter()

    for filename in sample:
        img_path = str((dataset_root / "photos" / filename).resolve())
        true_plate = normalize_gt(gt_data[filename])

        plate_image, _ = yolo_detect_single(img_path, best_model)
        if plate_image is None:
            continue

        plate_gray, plate_thresh = preprocess_plate_maciek_style(plate_image)

        pred = ""

        if PADDLE_AVAILABLE and paddle_reader is not None:
            raw1, final1 = read_plate_paddle(plate_gray)
            pred1 = normalize_gt(final1)

            if not pred1 or len(pred1) < 5:
                raw2, final2 = read_plate_paddle(plate_thresh)
                pred2 = normalize_gt(final2)
                pred = pred2 if pred2 else pred1
            else:
                pred = pred1
        else:
            gray_bgr = cv2.cvtColor(plate_gray, cv2.COLOR_GRAY2BGR)
            raw1, final1 = read_plate_easyocr(gray_bgr)
            pred1 = normalize_gt(final1)

            if not pred1 or len(pred1) < 5:
                thresh_bgr = cv2.cvtColor(plate_thresh, cv2.COLOR_GRAY2BGR)
                raw2, final2 = read_plate_easyocr(thresh_bgr)
                pred2 = normalize_gt(final2)
                pred = pred2 if pred2 else pred1
            else:
                pred = pred1

        acc = plate_accuracy_strict(pred, true_plate)
        total_acc += acc
        count += 1

    t1 = time.perf_counter()
    processing_time = t1 - t0
    mean_acc = total_acc / count if count > 0 else 0.0

    print("\n=== EWALUACJA PADDLE ===")
    print(f"Przetworzonych obrazów (z detekcją): {count}")
    print(f"Średnia accuracy:                  {mean_acc:.2f}%")
    print(f"Czas przetwarzania:                {processing_time:.2f} s")

    return mean_acc, processing_time


print("Multi-variant OCR ready")


# ============================================
# 6a. EasyOCR – główny OCR
# ============================================

def read_plate_easyocr(plate_bgr):
    """
    Stabilna wersja read_plate_easyocr (EasyOCR + proste reguły).
    """
    DIGIT_TO_LETTER = {
        "1": "I", "0": "O", "2": "Z", "3": "B", "4": "A", "5": "S",
        "6": "G", "7": "T", "8": "B", "9": "P"
    }

    INVALID_FIRST = {"1", "A", "H", "I", "J", "M", "Q", "U", "V", "X", "Y"}

    def run_easyocr_single(img_bgr):
        try:
            result = ocr_reader.readtext(img_bgr, detail=1, paragraph=False)
        except Exception:
            return ""

        if not result:
            return ""

        h = img_bgr.shape[0]
        cleaned = []

        for line in result:
            if len(line) < 3:
                continue
            box, txt, conf = line

            txt = txt.replace(" ", "").upper()
            txt = re.sub(r"[^A-Z0-9]", "", txt)
            if not txt:
                continue

            try:
                y_center = sum(p[1] for p in box) / 4.0
            except Exception:
                y_center = h / 2.0

            if y_center > h * 0.75:
                continue

            if len(txt) >= 2:
                cleaned.append((txt, float(conf)))

        if not cleaned:
            return ""

        raw_text = max(cleaned, key=lambda x: (len(x[0]), x[1]))[0]
        return raw_text

    def postprocess_text(text):
        if not text:
            return ""

        if text.startswith("PL"):
            text = text[2:]

        if len(text) > 0 and text[0] in INVALID_FIRST:
            text = text[1:]

        if len(text) > 8:
            text = text[:8]

        t = list(text)
        L = len(t)
        if L < 7:
            if len(t) >= 1 and t[0] in DIGIT_TO_LETTER:
                t[0] = DIGIT_TO_LETTER[t[0]]
        elif L < 8:
            for i in range(min(2, len(t))):
                if t[i] in DIGIT_TO_LETTER:
                    t[i] = DIGIT_TO_LETTER[t[i]]
        else:
            for i in range(min(3, len(t))):
                if t[i] in DIGIT_TO_LETTER:
                    t[i] = DIGIT_TO_LETTER[t[i]]

        if len(t) >= 2 and t[-1] == "0":
            t[-1] = "G"

        text = "".join(t)
        return text

    def score_candidate(s: str) -> int:
        if not s:
            return -999

        score = 0
        L = len(s)

        if 6 <= L <= 8:
            score += 4
        elif L == 5:
            score += 2
        else:
            score -= 4

        if L >= 1:
            if s[0].isalpha():
                score += 3
                if s[0] in {"S", "K"}:
                    score += 1
            else:
                score -= 2

        if L >= 2:
            if s[1].isalpha():
                score += 2
            else:
                score -= 1

        if L > 2:
            tail = s[2:]
            digits_tail = sum(c.isdigit() for c in tail)
            letters_tail = sum(c.isalpha() for c in tail)
            score += digits_tail
            if letters_tail > 2:
                score -= (letters_tail - 2) * 2

        return score

    if plate_bgr is None or plate_bgr.size == 0:
        return "", ""

    h, w = plate_bgr.shape[:2]
    if h < 8 or w < 20:
        return "", ""

    var_orig = plate_bgr

    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray_3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    _, thr = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)
    thr_3 = cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    clahe_3 = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)

    variants = [
        ("orig", var_orig),
        ("gray", gray_3),
        ("thr", thr_3),
        ("clahe", clahe_3),
    ]

    best_raw = ""
    best_final = ""
    best_score = -999

    for name, img_var in variants:
        raw = run_easyocr_single(img_var)
        if not raw:
            continue

        final = postprocess_text(raw)
        if not final:
            continue

        sc = score_candidate(final)
        if sc > best_score:
            best_score = sc
            best_raw = raw
            best_final = final

    if best_score == -999:
        return "", ""

    return best_raw, best_final


def read_plate_paddle(plate_image):
    """
    OCR tablicy z PaddleOCR:
    - czyści tekst,
    - filtruje dolne linie,
    - wybiera najdłuższy / najwyższy conf,
    - poprawia PL / pierwszą literę / cyfry->litery.
    """
    if paddle_reader is None:
        return "", ""

    DIGIT_TO_LETTER = {
        "1": "I", "0": "O", "2": "Z", "3": "B", "4": "A", "5": "S",
        "6": "G", "7": "T", "8": "B", "9": "P"
    }

    INVALID_FIRST = {"1", "A", "H", "I", "J", "M", "Q", "U", "V", "X", "Y"}

    result = paddle_reader.ocr(plate_image)

    if not result or not result[0]:
        return "", ""

    cleaned = []
    h = plate_image.shape[0]

    for line in result[0]:
        box = line[0]
        txt = line[1][0].replace(" ", "").upper()
        txt = re.sub(r"[^A-Z0-9]", "", txt)
        conf = float(line[1][1])

        y_center = sum(p[1] for p in box) / 4.0
        if y_center > h * 0.75:
            continue

        if len(txt) >= 2:
            cleaned.append((txt, conf))

    if not cleaned:
        return "", ""

    raw_text = max(cleaned, key=lambda x: (len(x[0]), x[1]))[0]
    text = raw_text

    if text.startswith("PL"):
        text = text[2:]

    if len(text) > 0 and text[0] in INVALID_FIRST:
        text = text[1:]

    if len(text) > 8:
        text = text[:8]

    t = list(text)
    if len(text) < 7:
        if len(t) >= 1 and t[0] in DIGIT_TO_LETTER:
            t[0] = DIGIT_TO_LETTER[t[0]]
    elif len(text) < 8:
        for i in range(min(2, len(t))):
            if t[i] in DIGIT_TO_LETTER:
                t[i] = DIGIT_TO_LETTER[t[i]]
    else:
        for i in range(min(3, len(t))):
            if t[i] in DIGIT_TO_LETTER:
                t[i] = DIGIT_TO_LETTER[t[i]]

    text = "".join(t)
    return raw_text, text


# ============================================
# 6b. Tesseract – fallback OCR
# ============================================

def read_plate_tesseract(plate_bgr):
    """
    Prosty Tesseract z whitelistą znaków (fallback).
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return "", ""

    try:
        gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        return "", ""

    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    _, thr = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    thr = cv2.resize(thr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    config = (
        "--oem 1 "
        "--psm 7 "
        "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
        "-c load_system_dawg=0 -c load_freq_dawg=0 "
    )

    try:
        text = pytesseract.image_to_string(thr, config=config)
    except Exception:
        return "", ""

    text = re.sub(r"[^A-Z0-9]", "", text.upper())
    return text, text


# ============================================
# 8. Ewaluacja – per obraz + dopasowanie IoU
# ============================================

def normalize_gt(text):
    if not text:
        return ""
    text = text.upper().replace(" ", "").replace("-", "")
    return re.sub(r"[^A-Z0-9]", "", text)


def match_detections_to_gt(gt_boxes, det_boxes, iou_thresh=0.5):
    """
    Greedy matching: zwraca listę par (gt_idx, det_idx, iou_val).
    """
    pairs = []
    unused_gt = set(range(len(gt_boxes)))
    unused_det = set(range(len(det_boxes)))

    all_iou = []
    for gi in unused_gt:
        for di in unused_det:
            all_iou.append((iou(gt_boxes[gi], det_boxes[di]), gi, di))
    all_iou.sort(reverse=True, key=lambda x: x[0])

    for iou_val, gi, di in all_iou:
        if iou_val < iou_thresh:
            break
        if gi in unused_gt and di in unused_det:
            pairs.append((gi, di, iou_val))
            unused_gt.remove(gi)
            unused_det.remove(di)

    return pairs, unused_gt, unused_det


def strip_prefix_noise(s: str) -> str:
    """
    Usuwa pojedynczy śmieciowy znak z przodu, jeśli poprawia score tablicy.
    """
    if not s:
        return s

    s = clean_text(s)

    if 8 <= len(s) <= 9:
        cand = s[1:]

        ok1, sc1 = validate_plate(s)
        ok2, sc2 = validate_plate(cand)

        if (ok2 and not ok1) or (ok2 and sc2 >= sc1):
            return cand

    return s


def evaluate_yolo_ocr(best_model: YOLO, val_records, max_images=100, iou_thresh=0.5):
    per_image = {}
    for r in val_records:
        per_image.setdefault(r["image_name"], []).append(r)

    image_names = list(per_image.keys())
    random.seed(42)
    random.shuffle(image_names)
    image_names = image_names[:max_images]

    correct_text = 0
    correct_det = 0
    total_gt = 0
    iou_values = []
    failures = []

    t0 = time.perf_counter()

    for img_name in image_names:
        recs = per_image[img_name]
        img_path = recs[0]["image_path"]

        gt_boxes = [tuple(map(int, r["bbox"])) for r in recs]
        gt_texts = [normalize_gt(r["plate_text"] or "") for r in recs]
        total_gt += len(gt_boxes)

        results = best_model.predict(
            source=img_path,
            imgsz=640,
            conf=0.25,
            device=DEVICE,
            verbose=False,
        )

        boxes = results[0].boxes
        det_boxes = []

        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls_id = int(b.cls[0].item())
                if cls_id != 0:
                    continue
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                det_boxes.append((x1, y1, x2, y2))

        if not det_boxes:
            for gt_txt in gt_texts:
                iou_values.append(0.0)
                failures.append((gt_txt, "", "no_detection"))
            continue

        pairs, unused_gt, unused_det = match_detections_to_gt(
            gt_boxes, det_boxes, iou_thresh=iou_thresh
        )

        for gi in unused_gt:
            iou_values.append(0.0)
            failures.append((gt_texts[gi], "", "no_detection"))

        img = cv2.imread(img_path)
        if img is None:
            for gi, di, iou_val in pairs:
                iou_values.append(iou_val)
                if iou_val >= iou_thresh:
                    correct_det += 1
                failures.append((gt_texts[gi], "", "img_read_fail"))
            continue

        h_img, w_img = img.shape[:2]

        for gi, di, iou_val in pairs:
            iou_values.append(iou_val)
            if iou_val >= iou_thresh:
                correct_det += 1

            gt_text = gt_texts[gi]
            det_box = det_boxes[di]

            x1, y1, x2, y2 = det_box
            w = x2 - x1
            h = y2 - y1

            pad_x = int(0.30 * w)
            pad_y = int(0.15 * h)

            x1p = max(0, x1 - pad_x)
            x2p = min(w_img - 1, x2 + pad_x)
            y1p = max(0, y1 - pad_y)
            y2p = min(h_img - 1, y2 + pad_y)

            crop = img[y1p:y2p, x1p:x2p]

            if crop.shape[0] < 10 or crop.shape[1] < 20:
                crop = img[y1:y2, x1:x2]

            crop = trim_left_pl_band(crop, pct=0.10)

            crop = cv2.resize(
                crop,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_CUBIC
            )

            e_raw, e_final = read_plate_easyocr(crop)
            e_pred = normalize_gt(e_final)

            t_raw, t_final = read_plate_tesseract(crop)
            t_pred = normalize_gt(t_final)

            candidates = []

            if e_pred:
                valid_e, score_e = validate_plate(e_pred)
                total_e = score_e + (10 if valid_e else 0)
                candidates.append((total_e, e_pred, "easy"))

            if t_pred:
                valid_t, score_t = validate_plate(t_pred)
                total_t = score_t + (10 if valid_t else 0)
                candidates.append((total_t, t_pred, "tess"))

            if not candidates:
                pred_text = ""
            else:
                candidates.sort(reverse=True, key=lambda x: x[0])
                pred_text = candidates[0][1]

            pred_text = strip_prefix_noise(pred_text)

            if pred_text == gt_text:
                correct_text += 1
            else:
                failures.append((gt_text, pred_text, "ocr_mismatch"))

    t1 = time.perf_counter()
    processing_time = t1 - t0

    det_accuracy = 100.0 * correct_det / total_gt if total_gt > 0 else 0.0
    ocr_accuracy = 100.0 * correct_text / total_gt if total_gt > 0 else 0.0
    mean_iou = float(np.mean(iou_values)) if iou_values else 0.0

    print(f"Detection accuracy (IoU >= {iou_thresh}): {det_accuracy:.2f}%")
    print(f"OCR accuracy (pełna tablica):           {ocr_accuracy:.2f}%")
    print(f"Średnie IoU:                            {mean_iou:.3f}")
    print(f"Czas przetwarzania {len(image_names)} zdjęć: {processing_time:.2f} s")

    print("\nPrzykładowe błędy OCR:")
    shown = 0
    for gt, pred, reason in failures:
        if reason == "ocr_mismatch":
            print(f"   GT: '{gt}' → PRED: '{pred}'")
            shown += 1
            if shown >= 15:
                break

    return det_accuracy, ocr_accuracy, processing_time, mean_iou


# ============================================
# 9. Ocena końcowa
# ============================================

def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0

    accuracy_norm = (accuracy_percent - 60) / 40
    time_norm = (60 - processing_time_sec) / 50

    score = 0.7 * accuracy_norm + 0.3 * time_norm
    grade = 2.0 + 3.0 * score
    return round(grade * 2) / 2


# ============================================
# 10. Główna funkcja
# ============================================

def main():
    dataset_root = get_dataset_root()
    usable_records = load_annotations(dataset_root)
    train_records, val_records = split_train_val(usable_records)

    print("Inicjalizuję EasyOCR...")
    print("EasyOCR initialized.")

    # show_samples(val_records, n=6)

    data_yaml = create_yolo_dataset(train_records, val_records, YOLO_ROOT)

    if TRAIN_YOLO:
        best_model = train_yolo_model(data_yaml, YOLO_ROOT)
    else:
        best_model = load_best_model(YOLO_ROOT)

    det_acc, ocr_acc, proc_time, mean_iou = evaluate_yolo_ocr(
        best_model, val_records, max_images=EVAL_MAX_IMAGES, iou_thresh=0.5
    )

    final_grade_strict = calculate_final_grade(ocr_acc, proc_time)

    print("\n--- PODSUMOWANIE (SUROWA METRYKA) ---")
    print(f"Detection accuracy: {det_acc:.2f}%")
    print(f"OCR accuracy:       {ocr_acc:.2f}%")
    print(f"Time for <=100 img: {proc_time:.2f} s")
    print(f"Średnie IoU:        {mean_iou:.3f}")
    print(f"Końcowa ocena (na podstawie OCR): {final_grade_strict}")

    paddle_acc, paddle_time = evaluate_paddle(
        best_model, dataset_root, max_images=EVAL_MAX_IMAGES
    )
    final_grade_paddle = calculate_final_grade(paddle_acc, paddle_time)

    print("\n--- PODSUMOWANIE (Paddle) ---")
    print(f"Mean accuracy (Paddle): {paddle_acc:.2f}%")
    print(f"Time for <=100 img:           {paddle_time:.2f} s")
    print(f"Końcowa ocena (Paddle): {final_grade_paddle}")


if __name__ == "__main__":
    main()
