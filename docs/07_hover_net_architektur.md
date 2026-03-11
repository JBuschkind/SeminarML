# HoVer-Net Architektur Dokumentation

## Übersicht

HoVer-Net ist eine **Encoder-Decoder Architektur** mit **Multi-Task Learning** für die Instanz-Segmentierung von Zellkernen.

## Architektur-Prinzipien

### 1. Encoder-Decoder Struktur

```
Input Image (3, H, W)
    ↓
┌─────────────────┐
│   Encoder       │  ← Extrahiert Features
│   (ResNet)      │
└────────┬────────┘
         │
    Features (C, H', W')
         │
┌────────▼────────┐
│   Decoder      │  ← Rekonstruiert Auflösung
│   (U-Net)      │
└────────┬────────┘
         │
    Multi-Task Outputs
```

### 2. Multi-Task Learning

Das Modell lernt gleichzeitig drei Aufgaben:

1. **Nuclear Segmentation**: Binäre Klassifikation (Kern vs. Hintergrund)
2. **HoVer Maps**: Regression (H/V-Vektoren)
3. **Type Classification**: Multi-Klassen-Klassifikation (optional)

**Vorteil**: Gemeinsame Features für alle Aufgaben → bessere Generalisierung

## Detaillierte Architektur

### Encoder (Backbone)

**Zweck**: Extrahiert semantische Features aus dem Bild

**Architektur**: ResNet oder ResNeXt

```
Input (3, H, W)
    ↓
Conv2d + BN + ReLU
    ↓
ResBlock 1 (stride=2) → (C1, H/2, W/2)
    ↓
ResBlock 2 (stride=2) → (C2, H/4, W/4)
    ↓
ResBlock 3 (stride=2) → (C3, H/8, W/8)
    ↓
ResBlock 4 (stride=2) → (C4, H/16, W/16)
```

**Features**:
- Verschiedene Auflösungen (Multi-Scale Features)
- Werden im Decoder verwendet

### Decoder (U-Net Style)

**Zweck**: Rekonstruiert volle Auflösung und kombiniert Multi-Scale Features

**Architektur**: U-Net mit Skip Connections

```
Encoder Features (C4, H/16, W/16)
    ↓
UpSample + Conv
    ↓
Concat mit Encoder Feature (C3, H/8, W/8)  ← Skip Connection
    ↓
UpSample + Conv
    ↓
Concat mit Encoder Feature (C2, H/4, W/4)  ← Skip Connection
    ↓
UpSample + Conv
    ↓
Concat mit Encoder Feature (C1, H/2, W/2)  ← Skip Connection
    ↓
UpSample + Conv
    ↓
Output (C_out, H, W)
```

**Skip Connections**: Verbinden Encoder- und Decoder-Features → behält räumliche Details

### Multi-Task Heads

Nach dem Decoder werden separate Heads für jede Aufgabe verwendet:

```
Shared Decoder Features (C, H, W)
    ↓
┌─────────────┬─────────────┬─────────────┐
│   Head 1    │   Head 2    │   Head 3    │
│  Nuclear    │   HoVer     │    Type     │
│ Segmentation│   Maps       │ Classification│
└─────────────┴─────────────┴─────────────┘
    ↓              ↓              ↓
(B, 1, H, W)  (B, 2, H, W)  (B, N, H, W)
```

**Head 1 - Nuclear Segmentation**:
```python
Conv2d(C, 32, 3, padding=1)
ReLU
Conv2d(32, 1, 1)
Sigmoid  # Binary output
```

**Head 2 - HoVer Maps**:
```python
Conv2d(C, 32, 3, padding=1)
ReLU
Conv2d(32, 2, 1)
Tanh  # Output [-1, 1]
```

**Head 3 - Type Classification** (optional):
```python
Conv2d(C, 32, 3, padding=1)
ReLU
Conv2d(32, N_types, 1)
Softmax  # Multi-class output
```

## Forward Pass

```python
def forward(self, x):
    # Encoder
    features = self.encoder(x)  # Multi-scale features
    
    # Decoder
    decoder_features = self.decoder(features)  # (B, C, H, W)
    
    # Multi-Task Heads
    nuclear_pred = self.nuclear_head(decoder_features)  # (B, 1, H, W)
    hover_pred = self.hover_head(decoder_features)     # (B, 2, H, W)
    type_pred = self.type_head(decoder_features)       # (B, N, H, W) optional
    
    return {
        'nuclear': nuclear_pred,
        'hover': hover_pred,
        'type': type_pred  # optional
    }
```

## Loss Functions

### Combined Loss

```python
total_loss = (
    λ1 * nuclear_loss(pred_nuclear, gt_nuclear) +
    λ2 * hover_loss(pred_hover, gt_hover) +
    λ3 * type_loss(pred_type, gt_type)  # optional
)
```

**Typische Werte**:
- `λ1 = 1.0` (Nuclear Segmentation)
- `λ2 = 1.0` (HoVer Maps)
- `λ3 = 0.5` (Type Classification, wenn verwendet)

### Einzelne Losses

**Nuclear Segmentation**: Binary Cross-Entropy oder Dice Loss
**HoVer Maps**: L1 oder L2 Loss
**Type Classification**: Cross-Entropy Loss

## Post-Processing

Nach der Vorhersage:

1. **Nuclear Segmentation**: Threshold (z.B. 0.5)
2. **HoVer Maps**: Werden für Instanz-Trennung verwendet
3. **Watershed**: Kombiniert Nuclear + HoVer für finale Instanzen

```python
# Pseudo-Code
nuclear_binary = (nuclear_pred > 0.5)
instances = watershed(nuclear_binary, hover_pred)
```

## Hyperparameter

### Architektur

- **Encoder**: ResNet34, ResNet50, oder ResNeXt50
- **Decoder Channels**: 256, 512, oder 1024
- **Input Size**: 256x256, 512x512, oder 1024x1024

### Training

- **Learning Rate**: 1e-4 bis 1e-3
- **Batch Size**: 4-16 (abhängig von GPU)
- **Optimizer**: Adam oder AdamW
- **Scheduler**: Cosine Annealing oder StepLR

## Implementierung

Die vollständige Implementierung finden Sie in:
- `src/models/hover_net.py` - Hauptmodell
- `src/models/losses.py` - Loss Functions

## Referenzen

- Original Paper: "HoVer-Net: Simultaneous Segmentation and Classification of Nuclei in Multi-Tissue Histology Images"
- Repository: https://github.com/vqdang/hover_net

## Nächste Schritte

- [Loss Functions](08_loss_functions.md) - Detaillierte Loss-Berechnung
- [Training Pipeline](09_training_pipeline.md) - Wie wird trainiert?
