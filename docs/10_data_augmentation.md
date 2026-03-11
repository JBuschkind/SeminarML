# Data Augmentation Dokumentation

## Übersicht

Data Augmentation erhöht die Robustheit des Modells durch Variation der Trainingsdaten.

## Warum Augmentation?

1. **Mehr Daten**: Erhöht effektive Datensatz-Größe
2. **Robustheit**: Model lernt invariant gegenüber Transformationen
3. **Overfitting**: Reduziert Overfitting
4. **Generalisierung**: Bessere Performance auf neuen Daten

## Verfügbare Augmentations

### 1. Horizontal Flip

**Was**: Spiegelt Bild horizontal

**Auf Masken**:
- Nuclear & Instance: Einfach gespiegelt
- HoVer: Horizontal-Komponente wird invertiert

```python
# Vorher: hover[:, :, 0] = [0.5, 0.3, ...]
# Nachher: hover[:, :, 0] = [-0.5, -0.3, ...]
```

### 2. Vertical Flip

**Was**: Spiegelt Bild vertikal

**Auf Masken**:
- Nuclear & Instance: Einfach gespiegelt
- HoVer: Vertikal-Komponente wird invertiert

### 3. Rotation

**Was**: Rotiert Bild um zufälligen Winkel

**Auf Masken**:
- Nuclear & Instance: Nearest-Neighbor Interpolation
- HoVer: Vektoren werden mit rotiert

**Rotation der Vektoren**:
```python
angle_rad = np.deg2rad(angle)
cos_a = np.cos(angle_rad)
sin_a = np.sin(angle_rad)

hover_new_h = hover_h * cos_a - hover_v * sin_a
hover_new_v = hover_h * sin_a + hover_v * cos_a
```

### 4. Scaling

**Was**: Skaliert Bild zufällig

**Auf Masken**:
- Nuclear & Instance: Nearest-Neighbor Interpolation
- HoVer: Lineare Interpolation

**Crop/Pad**: Nach Skalierung wird auf Originalgröße zurückgeschnitten/gepaddet

### 5. Color Jittering

**Was**: Ändert Helligkeit und Sättigung

**Nur auf Bildern**: Masken bleiben unverändert

**Implementierung**:
- Konvertierung zu HSV
- Zufällige Multiplikation von V (Brightness) und S (Saturation)
- Zurück zu RGB

### 6. Elastic Deformation (Optional)

**Was**: Nicht-lineare Verformung

**Status**: Noch nicht vollständig implementiert (langsam)

## Verwendung

### Train Augmentation

```python
from src.training.augmentations import get_train_augmentation

aug = get_train_augmentation(
    horizontal_flip=True,
    vertical_flip=True,
    rotation=True,
    rotation_range=(-15, 15),
    color_jitter=True,
    scale=True,
    scale_range=(0.9, 1.1)
)

# Im Dataset
dataset = NucleusDataset(..., transform=aug)
```

### Val Augmentation

```python
from src.training.augmentations import get_val_augmentation

aug = get_val_augmentation()  # Minimal oder keine Augmentation
```

### Custom Augmentation

```python
from src.training.augmentations import AugmentationPipeline

aug = AugmentationPipeline(
    horizontal_flip=True,
    vertical_flip=False,  # Nur horizontal
    rotation=False,  # Keine Rotation
    color_jitter=True,
    ...
)
```

## Wichtige Hinweise

### 1. Konsistenz

**Alle Komponenten müssen gleich transformiert werden**:
- Bild
- Nuclear Mask
- Instance Mask
- HoVer Map

### 2. HoVer Map Transformation

**Vektoren müssen korrekt transformiert werden**:
- Flip: Komponenten invertieren
- Rotation: Vektoren rotieren
- Scaling: Vektoren skalieren

### 3. Interpolation

**Masken**: Nearest-Neighbor (keine Blur-Effekte)
**Bilder**: Bilinear (glatte Transformationen)
**HoVer Maps**: Bilinear (Vektoren bleiben kontinuierlich)

## Beispiel

```python
# Original
data = {
    'image': image,      # (H, W, 3)
    'nuclear': nuclear,  # (H, W)
    'instance': instance, # (H, W)
    'hover': hover       # (H, W, 2)
}

# Augmentiert
augmented = aug(data)

# Alle haben gleiche Transformation
assert augmented['image'].shape == image.shape
assert augmented['nuclear'].shape == nuclear.shape
assert augmented['hover'].shape == hover.shape
```

## Performance

### Geschwindigkeit

- **Flip**: Sehr schnell
- **Rotation**: Schnell
- **Scaling**: Mittel
- **Color Jitter**: Schnell
- **Elastic Deformation**: Langsam

### Empfehlung

Für schnelles Training:
- Flip, Rotation, Color Jitter: ✅
- Scaling: Optional
- Elastic Deformation: ❌ (zu langsam)

## Best Practices

1. **Training**: Aggressive Augmentation
2. **Validation**: Keine oder minimale Augmentation
3. **Test**: Keine Augmentation
4. **HoVer Maps**: Immer korrekt transformieren
5. **Interpolation**: Nearest-Neighbor für Masken

## Nächste Schritte

- [Training Pipeline](09_training_pipeline.md) - Wie wird trainiert?
