# Evaluation Metrics Dokumentation

## Übersicht

Evaluation Metrics messen die Qualität der Vorhersagen des HoVer-Net Modells. Für Instanz-Segmentierung werden spezielle Metriken benötigt.

## Verfügbare Metrics

### 1. Dice Score

**Zweck**: Misst Übereinstimmung der binären Segmentierung

**Formel**:
```
Dice = (2 * |A ∩ B|) / (|A| + |B|)
```

**Bereich**: 0-1 (höher = besser)

**Verwendung**: Für Nuclear Segmentation (binär)

**Beispiel**:
```python
from src.evaluation.metrics import dice_score

dice = dice_score(pred_nuclear, target_nuclear)
print(f"Dice Score: {dice:.4f}")
```

### 2. Pixel Accuracy

**Zweck**: Anteil korrekt klassifizierter Pixel

**Formel**:
```
Accuracy = (Korrekte Pixel) / (Gesamte Pixel)
```

**Bereich**: 0-1 (höher = besser)

**Verwendung**: Allgemeine Pixel-Genauigkeit

### 3. Aggregated Jaccard Index (AJI)

**Zweck**: Hauptmetrik für Instanz-Segmentierung

**Was misst es?**:
- Wie gut sind Instanzen getrennt?
- Wie gut stimmen Instanzen mit Ground Truth überein?

**Berechnung**:
1. Matche vorhergesagte und Ground Truth Instanzen (basierend auf IoU)
2. Berechne Jaccard Index für gematchte Instanzen
3. Aggregiere über alle Instanzen

**Formel**:
```
AJI = (Summe Intersection gematchter Instanzen) / (Summe Union aller Instanzen)
```

**Bereich**: 0-1 (höher = besser)

**Typische Werte**:
- > 0.7: Sehr gut
- 0.5-0.7: Gut
- < 0.5: Verbesserung nötig

**Beispiel**:
```python
from src.evaluation.metrics import aggregated_jaccard_index

metrics = aggregated_jaccard_index(pred_instances, target_instances)
print(f"AJI: {metrics['aji']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1: {metrics['f1']:.4f}")
```

### 4. Panoptic Quality (PQ)

**Zweck**: Kombiniert Segmentierungs- und Detektions-Qualität

**Komponenten**:
- **Segmentation Quality (SQ)**: Durchschnittlicher IoU gematchter Instanzen
- **Detection Quality (DQ)**: Precision × Recall

**Formel**:
```
PQ = SQ × DQ
```

**Bereich**: 0-1 (höher = besser)

**Vorteil**: Berücksichtigt sowohl Segmentierung als auch Detektion

**Beispiel**:
```python
from src.evaluation.metrics import panoptic_quality

metrics = panoptic_quality(pred_instances, target_instances)
print(f"PQ: {metrics['pq']:.4f}")
print(f"SQ: {metrics['sq']:.4f}")
print(f"DQ: {metrics['dq']:.4f}")
```

### 5. Precision, Recall, F1

**Zweck**: Klassische Klassifikations-Metriken für Instanzen

**Berechnung**:
- **True Positives (TP)**: Korrekt detektierte Instanzen (IoU > Threshold)
- **False Positives (FP)**: Falsch detektierte Instanzen
- **False Negatives (FN)**: Übersehene Instanzen

**Formeln**:
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

## Verwendung

### Einzelne Vorhersage evaluieren

```python
from src.evaluation.metrics import evaluate_predictions

results = evaluate_predictions(
    nuclear_pred=nuclear_prediction,      # (H, W) in [0, 1]
    hover_pred=hover_prediction,          # (H, W, 2) in [-1, 1]
    nuclear_target=nuclear_ground_truth, # (H, W) in {0, 1}
    instance_target=instance_ground_truth, # (H, W) mit IDs
    threshold=0.5
)

print(f"Dice: {results['dice']:.4f}")
print(f"AJI: {results['aji']:.4f}")
print(f"PQ: {results['pq']:.4f}")
```

### Batch-Evaluation

```python
# Im Training/Evaluation Loop
all_metrics = []

for batch in dataloader:
    predictions = model(batch['image'])
    
    for i in range(batch_size):
        metrics = evaluate_predictions(
            predictions['nuclear'][i],
            predictions['hover'][i],
            batch['nuclear'][i],
            batch['instance'][i]
        )
        all_metrics.append(metrics)

# Durchschnitt berechnen
avg_dice = np.mean([m['dice'] for m in all_metrics])
avg_aji = np.mean([m['aji'] for m in all_metrics])
```

### Mit Evaluation Script

```bash
python scripts/evaluate.py \
    --config configs/config.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --split test \
    --save-predictions
```

## Instanz-Map Generierung

### Aus Predictions

```python
from src.evaluation.metrics import get_instance_map_from_predictions

instance_map = get_instance_map_from_predictions(
    nuclear_pred=nuclear_prediction,
    hover_pred=hover_prediction,
    threshold=0.5
)
```

**Wie es funktioniert**:
1. Binary Nuclear Mask (Threshold)
2. Distance Transform für Marker
3. Watershed Algorithm mit HoVer Maps als Elevation
4. Ergebnis: Instance Map mit eindeutigen IDs

## Interpretation der Ergebnisse

### Gute Werte

- **Dice > 0.8**: Sehr gute binäre Segmentierung
- **AJI > 0.7**: Sehr gute Instanz-Segmentierung
- **PQ > 0.6**: Gute Gesamtqualität
- **F1 > 0.8**: Gute Balance zwischen Precision und Recall

### Probleme erkennen

**Niedrige Dice, aber hohe AJI**:
- Binäre Segmentierung schlecht, aber Instanzen gut getrennt
- → Nuclear Segmentation Head verbessern

**Hohe Dice, aber niedrige AJI**:
- Binäre Segmentierung gut, aber Instanzen schlecht getrennt
- → HoVer Maps verbessern

**Niedrige Precision**:
- Zu viele False Positives
- → Threshold erhöhen oder Modell trainieren

**Niedrige Recall**:
- Zu viele False Negatives
- → Threshold senken oder mehr Daten

## Best Practices

1. **Threshold Tuning**: Optimalen Threshold für Nuclear Segmentation finden
2. **IoU Threshold**: Für AJI/PQ Matching (typisch 0.5)
3. **Mehrere Metriken**: Nicht nur eine Metrik verwenden
4. **Per-Sample Analyse**: Auch einzelne Samples analysieren

## Nächste Schritte

- [Visualisierung](12_visualisierung.md) - Wie werden Ergebnisse visualisiert?
