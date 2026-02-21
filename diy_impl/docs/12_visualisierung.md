# Visualisierung Dokumentation

## Übersicht

Das Visualisierungsmodul (`src/evaluation/visualizer.py`) erstellt Bilder wie im Beispiel: **Blaue Umrisse für Zellkerne**.

## Funktionen

### `visualize_annotations()`

Visualisiert Annotationen auf einem Bild.

**Parameter**:
- `image` (np.ndarray): Input-Bild (H, W, 3) RGB
- `masks` (Dict): Dictionary mit 'nuclear', 'instance', optional 'hover'
- `save_path` (str, optional): Pfad zum Speichern
- `alpha` (float): Transparenz (0-1)
- `show` (bool): Ob das Bild angezeigt werden soll

**Beispiel**:
```python
from src.evaluation.visualizer import visualize_annotations
import numpy as np

# Bild und Masken laden
image = np.array(...)  # (H, W, 3)
masks = {
    'nuclear': nuclear_mask,  # (H, W)
    'instance': instance_mask, # (H, W)
    'hover': hover_map         # (H, W, 2) optional
}

# Visualisieren
vis_image = visualize_annotations(
    image,
    masks,
    save_path='output.png',
    show=True
)
```

**Ausgabe**:
- Bild mit **blauen Umrissen** für jede Zelle
- Ähnlich dem Beispielbild

### `visualize_predictions()`

Vergleicht Ground Truth mit Vorhersagen.

**Beispiel**:
```python
from src.evaluation.visualizer import visualize_predictions

visualize_predictions(
    image,
    ground_truth={'nuclear': gt_nuclear, 'instance': gt_instance},
    prediction={'nuclear': pred_nuclear, 'instance': pred_instance},
    save_path='comparison.png'
)
```

**Ausgabe**: Drei Bilder nebeneinander:
1. Original
2. Ground Truth
3. Prediction

### `visualize_with_hover()`

Zeigt auch HoVer Maps.

**Beispiel**:
```python
from src.evaluation.visualizer import visualize_with_hover

visualize_with_hover(
    image,
    masks,
    save_path='hover_visualization.png'
)
```

**Ausgabe**: Vier Bilder:
1. Original
2. Annotiert
3. HoVer Map - Horizontal
4. HoVer Map - Vertical

## Farben

- **Blau**: Zellkerne (Nuclear Segmentation)
- **Grün**: Zellgrenzen (optional, wenn implementiert)
- **Rot**: Andere Strukturen (optional)

## Verwendung im Training

```python
# Nach Training
with torch.no_grad():
    predictions = model(images)
    
    # Visualisiere erste Vorhersage
    pred_nuclear = torch.sigmoid(predictions['nuclear'][0]).cpu().numpy()
    pred_hover = predictions['hover'][0].cpu().numpy()
    
    visualize_annotations(
        images[0].cpu().numpy().transpose(1, 2, 0),
        {
            'nuclear': (pred_nuclear > 0.5).astype(np.uint8),
            'hover': pred_hover.transpose(1, 2, 0)
        },
        save_path=f'outputs/predictions/epoch_{epoch}.png'
    )
```

## Nächste Schritte

- [Evaluation Metrics](11_evaluation_metrics.md) - Wie werden Ergebnisse gemessen?
