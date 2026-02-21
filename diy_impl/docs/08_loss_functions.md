# Loss Functions Dokumentation

## Übersicht

HoVer-Net verwendet **Multi-Task Learning** und benötigt daher kombinierte Loss Functions für verschiedene Aufgaben.

## Loss-Komponenten

### 1. Nuclear Segmentation Loss

**Aufgabe**: Binäre Klassifikation (Kern vs. Hintergrund)

**Optionen**:
- **Binary Cross-Entropy (BCE)**: Standard für binäre Klassifikation
- **Dice Loss**: Gut für unausgewogene Datensätze
- **Combined BCE + Dice**: Oft am besten

**Implementierung**:
```python
from src.models.losses import CombinedBCEDiceLoss

loss_fn = CombinedBCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
loss = loss_fn(pred_nuclear, target_nuclear)
```

### 2. HoVer Map Loss

**Aufgabe**: Regression von H/V-Vektoren

**Optionen**:
- **L1 Loss**: Robust gegen Ausreißer
- **L2 Loss (MSE)**: Glattere Gradienten

**Implementierung**:
```python
from src.models.losses import HoVerLoss

loss_fn = HoVerLoss(loss_type='l1')  # oder 'l2'
loss = loss_fn(pred_hover, target_hover, mask=nuclear_mask)
```

**Wichtig**: Loss wird nur auf Zellkern-Pixeln berechnet (mit Mask).

### 3. Type Classification Loss (Optional)

**Aufgabe**: Multi-Klassen-Klassifikation

**Loss**: Cross-Entropy Loss

**Implementierung**:
```python
loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
loss = loss_fn(pred_type, target_type)
```

## Combined Loss

### `HoVerNetLoss`

Kombiniert alle Loss-Komponenten:

```python
from src.models.losses import HoVerNetLoss

loss_fn = HoVerNetLoss(
    nuclear_weight=1.0,    # Gewicht für Nuclear Loss
    hover_weight=1.0,       # Gewicht für HoVer Loss
    type_weight=0.5,       # Gewicht für Type Loss (optional)
    hover_loss_type='l1',  # 'l1' oder 'l2'
    use_dice=True          # Ob Dice Loss verwendet wird
)

# Verwendung
predictions = {
    'nuclear': pred_nuclear,  # (B, 1, H, W)
    'hover': pred_hover,      # (B, 2, H, W)
    'type': pred_type         # (B, N, H, W) optional
}

targets = {
    'nuclear': target_nuclear,  # (B, H, W) oder (B, 1, H, W)
    'hover': target_hover,      # (B, 2, H, W)
    'type': target_type          # (B, H, W) optional
}

losses = loss_fn(predictions, targets)
total_loss = losses['total_loss']
```

**Rückgabe**:
```python
{
    'nuclear_loss': torch.Tensor,  # Nuclear segmentation loss
    'hover_loss': torch.Tensor,     # HoVer map loss
    'type_loss': torch.Tensor,      # Type classification loss
    'total_loss': torch.Tensor      # Combined loss
}
```

## Loss-Berechnung im Detail

### Nuclear Segmentation

```python
# Input: pred (B, 1, H, W) in [0, 1], target (B, H, W) in {0, 1}
# BCE Component
bce = -[target * log(pred) + (1-target) * log(1-pred)]

# Dice Component
intersection = (pred * target).sum()
union = pred.sum() + target.sum()
dice = (2 * intersection + smooth) / (union + smooth)
dice_loss = 1 - dice

# Combined
loss = 0.5 * bce + 0.5 * dice_loss
```

### HoVer Maps

```python
# Input: pred (B, 2, H, W) in [-1, 1], target (B, 2, H, W) in [-1, 1]
# Mask: nur auf Zellkern-Pixeln berechnen
mask = (nuclear_target > 0.5)

# L1 Loss
loss = |pred - target| * mask
loss = loss.sum() / mask.sum()
```

## Hyperparameter-Tuning

### Typische Gewichte

- `nuclear_weight = 1.0`: Hauptaufgabe
- `hover_weight = 1.0`: Wichtig für Instanz-Trennung
- `type_weight = 0.5`: Optional, weniger Gewicht

### Anpassung

Wenn eine Aufgabe dominiert:
- Erhöhen Sie das Gewicht der anderen Aufgaben
- Oder reduzieren Sie das Gewicht der dominierenden Aufgabe

## Verwendung im Training

```python
# Forward pass
predictions = model(images)

# Prepare targets
targets = {
    'nuclear': batch['nuclear'],
    'hover': batch['hover']
}

# Compute loss
losses = loss_fn(predictions, targets)
loss = losses['total_loss']

# Backward pass
loss.backward()
optimizer.step()

# Logging
print(f"Nuclear: {losses['nuclear_loss'].item():.4f}")
print(f"HoVer: {losses['hover_loss'].item():.4f}")
print(f"Total: {losses['total_loss'].item():.4f}")
```

## Nächste Schritte

- [Training Pipeline](09_training_pipeline.md) - Wie wird trainiert?
