# Inference Pipeline Dokumentation

## Übersicht

Die Inference Pipeline ermöglicht es, neue Bilder mit einem trainierten HoVer-Net Modell zu verarbeiten und Vorhersagen zu generieren.

## Verwendung

### Einfache Verwendung

```python
from src.utils.inference import HoVerNetInference

# Modell laden
inference = HoVerNetInference(
    checkpoint_path='outputs/checkpoints/best_model.pth',
    device='cuda',
    threshold=0.5
)

# Einzelnes Bild verarbeiten
predictions = inference.predict('path/to/image.tif')

print(f"Nuclear segmentation shape: {predictions['nuclear'].shape}")
print(f"Instance map shape: {predictions['instance'].shape}")
```

### Mit Command Line

```bash
# Einzelnes Bild
python scripts/inference.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --input path/to/image.tif \
    --output outputs/inference

# Mehrere Bilder (Verzeichnis)
python scripts/inference.py \
    --checkpoint outputs/checkpoints/best_model.pth \
    --input path/to/images/ \
    --output outputs/inference \
    --threshold 0.5
```

## API Referenz

### `HoVerNetInference`

Hauptklasse für Inference.

#### Initialisierung

```python
inference = HoVerNetInference(
    checkpoint_path: str,      # Pfad zum Checkpoint
    device: str = 'cuda',      # 'cuda' oder 'cpu'
    threshold: float = 0.5     # Threshold für Nuclear Segmentation
)
```

#### Methoden

**`predict(image, return_instances=True)`**

Verarbeitet ein einzelnes Bild.

**Parameter**:
- `image`: Bild (numpy array, PIL Image, oder Pfad)
- `return_instances`: Ob Instanz-Map generiert werden soll

**Rückgabe**:
```python
{
    'nuclear': np.ndarray,        # (H, W) in [0, 1]
    'hover': np.ndarray,          # (H, W, 2) in [-1, 1]
    'nuclear_binary': np.ndarray, # (H, W) in {0, 1}
    'instance': np.ndarray        # (H, W) mit eindeutigen IDs
}
```

**`process_image_file(image_path, output_dir, save_visualization=True)`**

Verarbeitet eine Bilddatei und speichert Ergebnisse.

**Parameter**:
- `image_path`: Pfad zur Bilddatei
- `output_dir`: Ausgabe-Verzeichnis
- `save_visualization`: Ob Visualisierung gespeichert werden soll

**Gespeicherte Dateien**:
- `{stem}_nuclear.png`: Nuclear Segmentation
- `{stem}_instances.png`: Instance Map (farbig)
- `{stem}_visualization.png`: Visualisierung (optional)

## Post-Processing

### Instanz-Map Generierung

Die Instanz-Map wird automatisch aus Nuclear Segmentation und HoVer Maps generiert:

1. **Binary Nuclear Mask**: Threshold auf Nuclear Prediction
2. **Distance Transform**: Für Marker-Generierung
3. **Watershed**: Mit HoVer Maps als Elevation
4. **Ergebnis**: Instance Map mit eindeutigen IDs

### Threshold Tuning

Der Threshold beeinflusst die Ergebnisse:

- **Niedriger Threshold (0.3-0.4)**: Mehr detektierte Zellen, aber möglicherweise mehr False Positives
- **Standard Threshold (0.5)**: Guter Kompromiss
- **Hoher Threshold (0.6-0.7)**: Weniger False Positives, aber möglicherweise mehr False Negatives

```python
# Mit anderem Threshold
inference = HoVerNetInference(
    checkpoint_path='model.pth',
    threshold=0.6  # Höherer Threshold
)
```

## Batch-Processing

### Mehrere Bilder

```python
images = ['image1.tif', 'image2.tif', 'image3.tif']
results = inference.predict_batch(images)
```

### Verzeichnis verarbeiten

```bash
python scripts/inference.py \
    --checkpoint model.pth \
    --input /path/to/images/ \
    --output /path/to/output/
```

## Visualisierung

### Automatische Visualisierung

Die Visualisierung wird automatisch gespeichert (blaue Umrisse für Zellkerne):

```python
predictions = inference.process_image_file(
    'image.tif',
    output_dir='outputs/',
    save_visualization=True
)
```

### Manuelle Visualisierung

```python
from src.evaluation.visualizer import visualize_annotations

# Predictions erhalten
predictions = inference.predict('image.tif')

# Visualisieren
visualize_annotations(
    image,
    {
        'nuclear': predictions['nuclear_binary'],
        'instance': predictions['instance']
    },
    save_path='output.png'
)
```

## Performance

### GPU vs. CPU

- **GPU**: Deutlich schneller (10-50x)
- **CPU**: Langsamer, aber universell verfügbar

### Optimierungen

1. **Batch Processing**: Mehrere Bilder gleichzeitig (wenn möglich)
2. **Image Size**: Kleinere Bilder = schneller
3. **Mixed Precision**: FP16 für schnellere GPU-Inference (optional)

## Beispiel-Workflow

### Komplettes Beispiel

```python
from src.utils.inference import HoVerNetInference
from pathlib import Path

# 1. Modell laden
inference = HoVerNetInference(
    checkpoint_path='outputs/checkpoints/best_model.pth',
    device='cuda'
)

# 2. Bild verarbeiten
image_path = Path('new_image.tif')
predictions = inference.process_image_file(
    image_path,
    output_dir='outputs/inference',
    save_visualization=True
)

# 3. Ergebnisse analysieren
num_instances = len(np.unique(predictions['instance'])) - 1  # Exclude background
print(f"Detected {num_instances} cell instances")

# 4. Weitere Verarbeitung
# z.B. Zellzählung, Größenanalyse, etc.
```

## Fehlerbehandlung

### Häufige Probleme

1. **Out of Memory**
   - Reduzieren Sie Bildgröße
   - Verwenden Sie CPU statt GPU
   - Batch-Größe reduzieren

2. **Falsche Bildgröße**
   - Wird automatisch behandelt
   - Model passt sich an

3. **Schlechte Predictions**
   - Threshold anpassen
   - Modell neu trainieren
   - Mehr Trainingsdaten

## Nächste Schritte

- [Evaluation Metrics](11_evaluation_metrics.md) - Wie werden Ergebnisse gemessen?
- [Visualisierung](12_visualisierung.md) - Wie werden Ergebnisse visualisiert?
