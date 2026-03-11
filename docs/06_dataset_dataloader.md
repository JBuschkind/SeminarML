# Dataset und DataLoader Dokumentation

## Übersicht

Das Dataset (`src/data/dataset.py`) und der DataLoader (`src/data/dataloader.py`) sind die Schnittstelle zwischen den rohen Daten und dem Training.

## PyTorch Dataset

### `NucleusDataset`

Eine PyTorch Dataset-Klasse, die Bilder und Masken lädt.

**Was macht sie?**
1. Findet zugehörige TIF- und XML-Dateien
2. Lädt Bilder
3. Generiert Masken aus XML-Annotationen (on-the-fly oder gecacht)
4. Wendet Transformationen an
5. Konvertiert zu PyTorch Tensors

### Initialisierung

```python
from src.data.dataset import NucleusDataset

dataset = NucleusDataset(
    data_dir="training_data",
    image_list=["TCGA-XX-XXXX-01Z-00-DX1_001", ...],
    transform=None,  # Optional: Data Augmentation
    cache_masks=False,  # Ob Masken im Speicher gecacht werden sollen
    generate_hover=True  # Ob HoVer Maps generiert werden sollen
)
```

**Parameter**:
- `data_dir` (str): Root-Verzeichnis des Datensatzes
- `image_list` (List[str]): Liste von Bildnamen (ohne Extension)
- `transform` (callable, optional): Transformationen für Bilder und Masken
- `cache_masks` (bool): Ob Masken gecacht werden sollen (spart Zeit, braucht RAM)
- `generate_hover` (bool): Ob HoVer Maps generiert werden sollen

### Verwendung

```python
# Dataset-Größe
print(f"Dataset size: {len(dataset)}")

# Einzelnes Sample laden
sample = dataset[0]
print(sample.keys())
# dict_keys(['image', 'nuclear', 'instance', 'hover', 'name'])

# Formate
print(f"Image: {sample['image'].shape}")      # (C, H, W) float32 [0, 1]
print(f"Nuclear: {sample['nuclear'].shape}")  # (H, W) int64
print(f"Instance: {sample['instance'].shape}") # (H, W) int64
print(f"HoVer: {sample['hover'].shape}")      # (2, H, W) float32
```

### Datenformat

**Input** (aus Dateien):
- Bilder: TIF-Format (RGB)
- Annotationen: XML-Format

**Output** (PyTorch Tensors):
```python
{
    'image': torch.Tensor,    # (C, H, W) float32, normalisiert [0, 1]
    'nuclear': torch.Tensor,  # (H, W) int64, Werte: 0 oder 1
    'instance': torch.Tensor, # (H, W) int64, Werte: 0, 1, 2, ...
    'hover': torch.Tensor,    # (2, H, W) float32, Werte: [-1, 1]
    'name': str               # Dateiname
}
```

## DataLoader

### Train/Val/Test Split

**Wichtig**: Split erfolgt auf **Proben-Ebene**, nicht auf Bild-Ebene!

**Warum?** Verhindert Data Leakage (Bilder derselben Probe sollten nicht in Train und Val/Test sein).

```python
from src.data.dataloader import create_train_val_test_split

splits = create_train_val_test_split(
    data_dir="training_data",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=42,
    save_path="data/splits/train_val_test_split.json"
)
```

**Output**:
```python
{
    'train': ['TCGA-XX-XXXX_001', 'TCGA-XX-XXXX_002', ...],
    'val': ['TCGA-YY-YYYY_001', ...],
    'test': ['TCGA-ZZ-ZZZZ_001', ...]
}
```

### DataLoader erstellen

```python
from src.data.dataloader import get_dataloaders

dataloaders = get_dataloaders(
    data_dir="training_data",
    batch_size=4,
    num_workers=4,
    split_file="data/splits/train_val_test_split.json",
    transform_train=my_train_transform,
    transform_val=my_val_transform,
    cache_masks=False,
    generate_hover=True
)
```

**Parameter**:
- `data_dir` (str): Root-Verzeichnis
- `batch_size` (int): Batch-Größe
- `num_workers` (int): Anzahl Worker-Prozesse (0 = kein Multi-Processing)
- `split_file` (str, optional): Pfad zu Split-JSON (wird erstellt, wenn nicht vorhanden)
- `transform_train` (callable, optional): Transformationen für Training
- `transform_val` (callable, optional): Transformationen für Validation/Test
- `cache_masks` (bool): Ob Masken gecacht werden sollen
- `generate_hover` (bool): Ob HoVer Maps generiert werden sollen

**Rückgabe**:
```python
{
    'train': DataLoader,
    'val': DataLoader,
    'test': DataLoader
}
```

### Verwendung im Training

```python
for epoch in range(num_epochs):
    # Training
    for batch in dataloaders['train']:
        images = batch['image']      # (B, C, H, W)
        nuclear = batch['nuclear']    # (B, H, W)
        instance = batch['instance']  # (B, H, W)
        hover = batch['hover']        # (B, 2, H, W)
        
        # Forward pass
        predictions = model(images)
        
        # Loss calculation
        loss = compute_loss(predictions, nuclear, hover)
        
        # Backward pass
        loss.backward()
        optimizer.step()
    
    # Validation
    with torch.no_grad():
        for batch in dataloaders['val']:
            # ...
```

## Collate Function

### Problem: Variable Bildgrößen

Bilder im Datensatz haben unterschiedliche Größen:
- Bild 1: (3, 157, 185)
- Bild 2: (3, 512, 512)

`torch.stack()` kann das nicht handhaben!

### Lösung: Padding

Die `collate_fn` Funktion:
1. Findet die maximale Größe im Batch
2. Paddet alle Bilder/Masken auf diese Größe
3. Stapelt sie dann

```python
def collate_fn(batch):
    # Finde maximale Dimensionen
    max_h = max(item['image'].shape[1] for item in batch)
    max_w = max(item['image'].shape[2] for item in batch)
    
    # Pad alle Tensoren
    padded_images = [pad_to_size(img, (max_h, max_w)) for img in images]
    
    # Stack
    return {
        'image': torch.stack(padded_images),
        'nuclear': torch.stack(padded_nuclear),
        ...
    }
```

**Padding-Werte**:
- Bilder: `0.0` (schwarz)
- Masken: `0` (Hintergrund)

## Caching

### Masken-Caching

```python
dataset = NucleusDataset(..., cache_masks=True)
```

**Vorteile**:
- Schneller (Masken werden nur einmal generiert)
- Nützlich bei wiederholtem Zugriff

**Nachteile**:
- Braucht mehr RAM
- Nicht für sehr große Datensätze geeignet

**Empfehlung**: 
- `True` für kleine Datensätze oder wenn RAM verfügbar ist
- `False` für große Datensätze

## Transformationen

### Beispiel: Data Augmentation

```python
def train_transform(data):
    image = data['image']
    nuclear = data['nuclear']
    instance = data['instance']
    hover = data['hover']
    
    # Random horizontal flip
    if random.random() > 0.5:
        image = np.fliplr(image)
        nuclear = np.fliplr(nuclear)
        instance = np.fliplr(instance)
        hover = np.fliplr(hover)
        hover[:, :, 0] *= -1  # Invertiere Horizontal-Komponente
    
    # Random rotation
    angle = random.uniform(-15, 15)
    # ... rotation code ...
    
    return {
        'image': image,
        'nuclear': nuclear,
        'instance': instance,
        'hover': hover
    }
```

**Wichtig**: Transformationen müssen auf **alle** Komponenten angewendet werden (Bild + alle Masken)!

## Performance-Tipps

1. **num_workers**: 
   - Windows: Oft `0` (Multi-Processing Probleme)
   - Linux/Mac: `4-8` für gute Performance

2. **pin_memory**: 
   - `True` wenn GPU verwendet wird
   - `False` wenn nur CPU

3. **prefetch_factor**: 
   - Standard: `2`
   - Erhöhen für bessere GPU-Auslastung

## Fehlerbehandlung

### Häufige Probleme

1. **Dateien nicht gefunden**
   - Überprüfen Sie `data_dir` und `image_list`
   - Stellen Sie sicher, dass TIF- und XML-Dateien existieren

2. **Out of Memory**
   - Reduzieren Sie `batch_size`
   - Setzen Sie `cache_masks=False`

3. **Variable Größen**
   - Wird automatisch durch `collate_fn` behandelt
   - Kann Performance beeinträchtigen (Padding)

## Nächste Schritte

- [HoVer-Net Architektur](07_hover_net_architektur.md) - Wie ist das Modell aufgebaut?
