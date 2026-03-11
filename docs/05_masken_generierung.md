# Masken-Generierung Dokumentation

## Übersicht

Der Masken-Generator (`src/data/mask_generator.py`) konvertiert Polygon-Annotationen in die Masken-Formate, die HoVer-Net benötigt.

## Warum verschiedene Masken?

HoVer-Net verwendet **Multi-Task Learning** und benötigt daher verschiedene Masken-Typen:

1. **Nuclear Segmentation Map**: Binärmaske - "Wo sind Zellkerne?"
2. **Instance Map**: Eindeutige IDs - "Welche Zelle ist welche?"
3. **HoVer Map**: H/V-Vektoren - "Wie trenne ich überlappende Zellen?"

## Masken-Typen im Detail

### 1. Nuclear Segmentation Map

**Format**: `(H, W)` - Binary (0 oder 1)

**Bedeutung**:
- `1` = Pixel gehört zu einem Zellkern
- `0` = Hintergrund

**Verwendung**: 
- Binary Classification Loss
- Zeigt dem Modell, wo Zellkerne sind

**Beispiel**:
```python
nuclear_mask = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
])
```

### 2. Instance Map

**Format**: `(H, W)` - Integer (0, 1, 2, 3, ...)

**Bedeutung**:
- `0` = Hintergrund
- `1` = Erste Zelle
- `2` = Zweite Zelle
- `3` = Dritte Zelle
- etc.

**Verwendung**:
- Für Post-Processing (Watershed)
- Evaluation (Instanz-Metriken)
- Visualisierung

**Beispiel**:
```python
instance_map = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 2, 0],
    [0, 1, 1, 2, 0],
    [0, 0, 0, 0, 0]
])
# Zwei Zellen: ID 1 und ID 2
```

### 3. HoVer Map

**Format**: `(H, W, 2)` - Float (Horizontal, Vertical)

**Bedeutung**:
- Für jeden Pixel in einer Zelle: Vektor zum nächsten Rand
- `[:, :, 0]` = Horizontal-Komponente (-1 bis +1)
- `[:, :, 1]` = Vertikal-Komponente (-1 bis +1)

**Verwendung**:
- Hilft dem Modell, überlappende Zellen zu trennen
- Wird für Instanz-Separation verwendet

**Beispiel**:
```python
hover_map = np.array([
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [-0.7, -0.7], [0.7, -0.7]],  # Vektoren zum Rand
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
])
```

## API Referenz

### `generate_masks(regions, image_shape, generate_hover=True, generate_type=False)`

Generiert alle Masken aus Polygon-Regionen.

**Parameter**:
- `regions` (List[Dict]): Liste von Regionen mit 'vertices' (Nx2 Array)
- `image_shape` (Tuple[int, int]): (height, width) der Ausgabe-Masken
- `generate_hover` (bool): Ob HoVer Maps generiert werden sollen
- `generate_type` (bool): Ob Type Maps generiert werden sollen (noch nicht implementiert)

**Rückgabe** (Dict):
```python
{
    'nuclear': np.ndarray,      # (H, W) uint8
    'instance': np.ndarray,     # (H, W) int32
    'hover': np.ndarray,       # (H, W, 2) float32 (optional)
}
```

**Beispiel**:
```python
from src.data.mask_generator import generate_masks

regions = [
    {'vertices': np.array([[10, 10], [20, 10], [20, 20], [10, 20]])},
    {'vertices': np.array([[30, 30], [40, 30], [40, 40], [30, 40]])},
]

masks = generate_masks(regions, (50, 50), generate_hover=True)

print(f"Nuclear mask shape: {masks['nuclear'].shape}")
print(f"Instance mask shape: {masks['instance'].shape}")
print(f"HoVer map shape: {masks['hover'].shape}")
```

## Implementierungs-Details

### 1. Polygon-Füllung

```python
from skimage.draw import polygon

rr, cc = polygon(vertices[:, 1], vertices[:, 0], shape=(height, width))
nuclear_mask[rr, cc] = 1
instance_mask[rr, cc] = instance_id
```

**Wie es funktioniert**:
- `polygon()` gibt alle Pixel-Koordinaten innerhalb des Polygons zurück
- Diese werden in der Maske auf 1 gesetzt

### 2. HoVer Map Generierung

```python
def generate_hover_maps_optimized(instance_mask, nuclear_mask):
    # Für jede Instanz:
    for instance_id in instance_ids:
        instance_binary = (instance_mask == instance_id)
        
        # Distance Transform: Abstand zum Rand
        dist_transform = distance_transform_edt(instance_binary)
        
        # Gradient: Richtung zum Rand
        sobel_x = cv2.Sobel(dist_transform, ...)  # Horizontal
        sobel_y = cv2.Sobel(dist_transform, ...)  # Vertikal
        
        # Normalisieren und invertieren
        hover_map[instance_mask, 0] = -sobel_x[instance_mask]
        hover_map[instance_mask, 1] = -sobel_y[instance_mask]
```

**Wie es funktioniert**:
1. **Distance Transform**: Berechnet für jeden Pixel den Abstand zum nächsten Rand
2. **Gradient**: Berechnet die Richtung (zeigt weg vom Rand)
3. **Invertieren**: Kehrt die Richtung um (zeigt zum Rand)
4. **Normalisieren**: Macht Vektoren zu Einheitsvektoren

**Warum?** 
- Zellen in der Mitte haben große Vektoren (weit vom Rand)
- Zellen am Rand haben kleine Vektoren (nah am Rand)
- Dies hilft dem Modell, Instanzen zu trennen

## Visualisierung

```python
import matplotlib.pyplot as plt

# Nuclear mask
plt.imshow(masks['nuclear'], cmap='gray')
plt.title('Nuclear Segmentation')

# Instance map
plt.imshow(masks['instance'], cmap='nipy_spectral')
plt.title('Instance Map')

# HoVer map
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(masks['hover'][:, :, 0], cmap='RdBu')
ax1.set_title('HoVer - Horizontal')
ax2.imshow(masks['hover'][:, :, 1], cmap='RdBu')
ax2.set_title('HoVer - Vertical')
```

## Performance

- **Polygon-Füllung**: Sehr schnell (skimage)
- **HoVer Maps**: Kann bei vielen Instanzen langsam sein
- **Optimierung**: Verwendet optimierte Version mit Distance Transform

## Fehlerbehandlung

### Häufige Probleme

1. **Vertices außerhalb Bildgrenzen**
   ```python
   vertices[:, 0] = np.clip(vertices[:, 0], 0, width - 1)
   ```
   **Lösung**: Automatisches Clipping

2. **Leere Regionen**
   - Wird übersprungen
   - Keine Auswirkung auf Maske

3. **Sehr kleine Regionen**
   - Werden trotzdem erstellt
   - Können bei Evaluation gefiltert werden

## Nächste Schritte

- [Dataset und DataLoader](06_dataset_dataloader.md) - Wie werden Masken geladen?
