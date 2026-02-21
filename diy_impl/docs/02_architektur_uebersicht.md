# Architektur-Übersicht

## Gesamtarchitektur

Das Projekt ist in mehrere Module unterteilt, die zusammenarbeiten:

```
┌─────────────────────────────────────────────────────────┐
│                    Input Layer                           │
│  Histopathologie-Bilder (TIF) + XML-Annotationen        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Datenverarbeitung (src/data/)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ XML-Parser   │→ │ Masken-      │→ │ Dataset &    │  │
│  │              │  │ Generator    │  │ DataLoader   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Modell-Architektur (src/models/)              │
│  ┌──────────────────────────────────────────────────┐  │
│  │              HoVer-Net Modell                     │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐      │  │
│  │  │ Encoder   │→ │ Decoder   │→ │ Multi-   │      │  │
│  │  │ (ResNet)  │  │ (U-Net)   │  │ Task     │      │  │
│  │  └──────────┘  └──────────┘  │ Outputs   │      │  │
│  │                               └──────────┘      │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│            Training Pipeline (src/training/)            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Data         │→ │ Loss         │→ │ Optimizer    │  │
│  │ Augmentation │  │ Functions    │  │ & Scheduler  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          Evaluation & Visualisierung                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Metrics       │  │ Visualizer  │  │ Post-       │  │
│  │ (Dice, AJI)   │  │             │  │ Processing   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Modul-Übersicht

### 1. Datenverarbeitung (`src/data/`)

**Zweck**: Konvertiert rohe Daten (Bilder + XML) in trainierbare Formate

**Komponenten**:
- `xml_parser.py`: Parst XML-Annotationen
- `mask_generator.py`: Erstellt Masken (Nuclear, Instance, HoVer)
- `dataset.py`: PyTorch Dataset-Klasse
- `dataloader.py`: DataLoader mit Train/Val/Test Split

**Input**: TIF-Bilder + XML-Dateien
**Output**: PyTorch Tensors (Bilder + Masken)

### 2. Modell-Architektur (`src/models/`)

**Zweck**: Definiert die HoVer-Net Architektur

**Komponenten**:
- `hover_net.py`: Hauptmodell (Encoder-Decoder)
- `losses.py`: Loss Functions (Multi-Task)

**Input**: Bilder (B, C, H, W)
**Output**: 
- Nuclear Segmentation (B, 1, H, W)
- HoVer Maps (B, 2, H, W)
- Type Classification (B, N_types, H, W) - optional

### 3. Training (`src/training/`)

**Zweck**: Training-Logik und -Utilities

**Komponenten**:
- `trainer.py`: Training Loop
- `augmentations.py`: Data Augmentation

**Features**:
- Checkpointing
- Logging (TensorBoard)
- Early Stopping
- Learning Rate Scheduling

### 4. Evaluation (`src/evaluation/`)

**Zweck**: Bewertung und Visualisierung der Ergebnisse

**Komponenten**:
- `metrics.py`: Evaluation Metrics
- `visualizer.py`: Visualisierungstools

**Metrics**:
- Dice Score
- Aggregated Jaccard Index (AJI)
- Panoptic Quality (PQ)

## Datenfluss im Detail

### Phase 1: Datenaufbereitung

```
XML-Datei
  ↓ parse_xml_annotations()
Polygon-Vertices (Liste von [x, y] Koordinaten)
  ↓ generate_masks()
┌─────────────────────────────────────┐
│ Nuclear Map:      (H, W) binary    │
│ Instance Map:     (H, W) int32      │
│ HoVer Map:        (H, W, 2) float  │
└─────────────────────────────────────┘
```

### Phase 2: Training

```
Batch von Bildern
  ↓
HoVer-Net Modell
  ↓
┌─────────────────────────────────────┐
│ Nuclear Prediction:  (B, 1, H, W)  │
│ HoVer Prediction:    (B, 2, H, W)  │
│ Type Prediction:     (B, N, H, W)  │
└─────────────────────────────────────┘
  ↓
Loss Functions
  ↓
Backpropagation & Update
```

### Phase 3: Inference

```
Neues Bild
  ↓
HoVer-Net (inference mode)
  ↓
Predictions
  ↓
Post-Processing (Watershed, etc.)
  ↓
Instanz-Segmentierung
  ↓
Visualisierung (blaue Umrisse)
```

## Design-Prinzipien

1. **Modularität**: Jedes Modul hat eine klare Verantwortung
2. **Wiederverwendbarkeit**: Komponenten können unabhängig verwendet werden
3. **Erweiterbarkeit**: Neue Features können einfach hinzugefügt werden
4. **Dokumentation**: Alles ist ausführlich dokumentiert

## Nächste Schritte

- [XML-Parser](04_xml_parser.md) - Wie werden Annotationen geparst?
- [Masken-Generierung](05_masken_generierung.md) - Wie entstehen die Masken?
