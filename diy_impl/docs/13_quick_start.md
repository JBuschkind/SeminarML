# Quick Start Guide

## Schnellstart in 5 Minuten

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Dataloader testen

```bash
python scripts/test_dataloader.py
```

Dies erstellt:
- Train/Val/Test Split
- Testet den DataLoader
- Generiert Beispiel-Visualisierungen

### 3. Daten verstehen

```python
from src.data.dataloader import get_dataloaders

# DataLoader erstellen
dataloaders = get_dataloaders(
    data_dir="training_data",
    batch_size=4,
    num_workers=0  # Windows: 0, Linux/Mac: 4
)

# Ein Batch laden
batch = next(iter(dataloaders['train']))
print(f"Image shape: {batch['image'].shape}")      # (B, C, H, W)
print(f"Nuclear shape: {batch['nuclear'].shape}")  # (B, H, W)
print(f"HoVer shape: {batch['hover'].shape}")     # (B, 2, H, W)
```

### 4. Modell erstellen (nach Implementierung)

```python
from src.models.hover_net import HoVerNet

model = HoVerNet(
    num_types=4,  # Anzahl Zelltypen (optional)
    input_size=512
)
```

### 5. Training starten (nach Implementierung)

```bash
python scripts/train.py --config configs/config.yaml
```

## Nächste Schritte

- Lesen Sie die [vollständige Dokumentation](README.md)
- Verstehen Sie die [Architektur](02_architektur_uebersicht.md)
- Schauen Sie sich [Beispiele](15_beispiele.md) an
