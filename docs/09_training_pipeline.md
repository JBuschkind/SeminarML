# Training Pipeline Dokumentation

## Übersicht

Die Training Pipeline (`src/training/`) verwaltet den gesamten Trainingsprozess: Training Loop, Validation, Checkpointing und Logging.

## Komponenten

### 1. Trainer (`trainer.py`)

Die Hauptklasse für das Training.

**Features**:
- Training Loop mit Progress Bars
- Validation Loop
- Checkpointing (Best & Last Model)
- TensorBoard Logging
- Learning Rate Scheduling Support

### 2. Data Augmentation (`augmentations.py`)

Augmentations für robusteres Training.

**Verfügbare Augmentations**:
- Horizontal/Vertical Flip
- Rotation
- Scaling
- Color Jittering
- Elastic Deformation (optional)

## Verwendung

### Einfaches Training

```python
from src.models import HoVerNet, HoVerNetLoss
from src.data.dataloader import get_dataloaders
from src.training.trainer import Trainer
from src.training.augmentations import get_train_augmentation, get_val_augmentation
import torch.optim as optim

# Model erstellen
model = HoVerNet(backbone='resnet34', pretrained=True)

# Data Loaders
train_aug = get_train_augmentation()
val_aug = get_val_augmentation()
dataloaders = get_dataloaders(
    data_dir="training_data",
    transform_train=train_aug,
    transform_val=val_aug
)

# Loss & Optimizer
loss_fn = HoVerNetLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Trainer
trainer = Trainer(
    model=model,
    train_loader=dataloaders['train'],
    val_loader=dataloaders['val'],
    loss_fn=loss_fn,
    optimizer=optimizer,
    device='cuda'
)

# Training starten
trainer.train(num_epochs=100)
```

### Mit Konfigurationsdatei

```bash
python scripts/train.py --config configs/config.yaml
```

### Training fortsetzen

```bash
python scripts/train.py --config configs/config.yaml --resume outputs/checkpoints/last_model.pth
```

## Konfiguration

### Config-Datei (`configs/config.yaml`)

```yaml
# Model
model:
  backbone: "resnet34"
  pretrained: true
  decoder_channels: 256

# Training
training:
  num_epochs: 100
  learning_rate: 0.0001
  optimizer: "adam"
  scheduler: "cosine"

# Augmentation
augmentation:
  train:
    horizontal_flip: true
    rotation: true
    ...
```

## Training Loop

### Ablauf

1. **Epoch Start**
   - Model in Training Mode
   - Progress Bar initialisieren

2. **Training Loop**
   - Batch laden
   - Forward Pass
   - Loss berechnen
   - Backward Pass
   - Optimizer Step
   - Logging (alle N Batches)

3. **Validation** (alle N Epochs)
   - Model in Eval Mode
   - Validation Loop
   - Loss berechnen
   - Best Model speichern (wenn besser)

4. **Checkpointing**
   - Best Model speichern
   - Last Model speichern

5. **Logging**
   - TensorBoard Logs schreiben
   - Console Output

## Checkpointing

### Gespeicherte Informationen

```python
checkpoint = {
    'epoch': epoch,
    'global_step': step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_loss': best_loss,
    'train_losses': train_losses,
    'val_losses': val_losses
}
```

### Checkpoint laden

```python
trainer.load_checkpoint('outputs/checkpoints/best_model.pth')
```

## Logging

### TensorBoard

```bash
tensorboard --logdir outputs/logs
```

**Geloggte Metriken**:
- Train/BatchLoss
- Train/EpochLoss
- Train/NuclearLoss
- Train/HoverLoss
- Val/TotalLoss
- Val/NuclearLoss
- Val/HoverLoss
- Train/LearningRate

### Console Output

```
Epoch 1/100
Train Loss: 0.8234 (Nuclear: 0.5123, HoVer: 0.3111)
Val Loss: 0.7890 (Nuclear: 0.4987, HoVer: 0.2903)
Saved best model (val_loss: 0.7890)
Epoch time: 45.23s
```

## Data Augmentation

### Train Augmentation

```python
train_aug = get_train_augmentation(
    horizontal_flip=True,
    vertical_flip=True,
    rotation=True,
    rotation_range=(-15, 15),
    color_jitter=True,
    scale=True,
    scale_range=(0.9, 1.1)
)
```

**Wichtig**: Alle Augmentations werden auf Bilder UND Masken angewendet!

### Val Augmentation

```python
val_aug = get_val_augmentation()  # Minimal oder keine Augmentation
```

## Learning Rate Scheduling

### Cosine Annealing

```python
scheduler = CosineAnnealingLR(optimizer, T_max=100)
```

### Step LR

```python
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
```

## Best Practices

1. **Validation**: Regelmäßig validieren (jedes Epoch)
2. **Checkpointing**: Bestes und letztes Model speichern
3. **Logging**: TensorBoard für Visualisierung
4. **Augmentation**: Nur beim Training, nicht bei Validation
5. **Early Stopping**: Implementieren wenn Overfitting auftritt

## Troubleshooting

### Out of Memory

- Reduzieren Sie `batch_size`
- Setzen Sie `cache_masks=False`
- Verwenden Sie kleinere Bilder

### Training zu langsam

- Erhöhen Sie `num_workers` (nicht auf Windows)
- Reduzieren Sie `log_interval`
- Verwenden Sie GPU

### Loss steigt nicht

- Überprüfen Sie Learning Rate
- Überprüfen Sie Data Augmentation
- Überprüfen Sie Loss Weights

## Nächste Schritte

- [Evaluation Metrics](11_evaluation_metrics.md) - Wie werden Ergebnisse gemessen?
