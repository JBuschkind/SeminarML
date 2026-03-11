# Installation und Setup

## Systemanforderungen

- **Python**: 3.8 oder höher
- **Betriebssystem**: Windows, Linux, oder macOS
- **RAM**: Mindestens 8 GB (16 GB empfohlen)
- **GPU**: Optional, aber empfohlen für Training (CUDA-kompatibel)

## Schritt-für-Schritt Installation

### 1. Repository klonen/öffnen

```bash
cd D:\Git\SeminarML\diy_impl
```

### 2. Virtual Environment erstellen (empfohlen)

**Windows**:
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac**:
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Dependencies installieren

```bash
pip install -r requirements.txt
```

**Wichtig**: Wenn Sie eine GPU verwenden möchten, installieren Sie PyTorch mit CUDA-Unterstützung:

```bash
# Für CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Für CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Dann andere Dependencies
pip install -r requirements.txt
```

### 4. Verzeichnisse erstellen

```bash
# Windows PowerShell
New-Item -ItemType Directory -Force -Path data\splits, outputs\checkpoints, outputs\logs, outputs\predictions

# Linux/Mac
mkdir -p data/splits outputs/{checkpoints,logs,predictions}
```

### 5. Testen der Installation

```bash
python scripts/test_dataloader.py
```

Wenn alles funktioniert, sollten Sie sehen:
- Train/Val/Test Split wird erstellt
- DataLoader lädt Daten
- Visualisierungen werden gespeichert

## Verifizierung

### Python-Version prüfen

```python
python --version
# Sollte 3.8 oder höher sein
```

### Pakete prüfen

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

### GPU-Verfügbarkeit prüfen

```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Wenn `True`, können Sie die GPU verwenden.

## Projektstruktur nach Installation

```
diy_impl/
├── venv/                          # Virtual Environment
├── training_data/ # Datensatz
├── src/                            # Quellcode
├── scripts/                        # Scripts
├── data/
│   └── splits/                    # Train/Val/Test Splits
├── outputs/
│   ├── checkpoints/               # Model Checkpoints
│   ├── logs/                      # Training Logs
│   └── predictions/               # Vorhersagen
└── docs/                          # Dokumentation
```

## Häufige Probleme

### Problem 1: ModuleNotFoundError

**Symptom**: `ModuleNotFoundError: No module named 'numpy'`

**Lösung**:
```bash
pip install -r requirements.txt
```

### Problem 2: OpenCV libGL.so.1 Fehler (Linux)

**Symptom**: `ImportError: libGL.so.1: cannot open shared object file`

**Lösung**: Verwenden Sie `opencv-python-headless` statt `opencv-python`:
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

Oder installieren Sie `requirements.txt` neu (enthält bereits headless Version).

### Problem 3: CUDA nicht verfügbar

**Symptom**: `torch.cuda.is_available()` gibt `False` zurück

**Lösungen**:
1. Überprüfen Sie, ob CUDA installiert ist: `nvidia-smi`
2. Installieren Sie PyTorch mit CUDA-Unterstützung (siehe oben)
3. Falls keine GPU vorhanden: Training funktioniert auch auf CPU (langsamer)

### Problem 4: Windows Multi-Processing Fehler

**Symptom**: `RuntimeError` bei DataLoader mit `num_workers > 0`

**Lösung**: Setzen Sie `num_workers=0` in den DataLoader-Aufrufen

```python
dataloaders = get_dataloaders(..., num_workers=0)
```

### Problem 5: Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Lösungen**:
1. Reduzieren Sie `batch_size`
2. Setzen Sie `cache_masks=False`
3. Verwenden Sie kleinere Bildgrößen

## Nächste Schritte

Nach erfolgreicher Installation:
1. [Quick Start Guide](13_quick_start.md) - Erste Schritte
2. [Dataset und DataLoader](06_dataset_dataloader.md) - Daten verstehen
