# Troubleshooting Guide

## Häufige Probleme und Lösungen

### 1. OpenCV libGL.so.1 Fehler

**Fehler**:
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

**Ursache**: `opencv-python` benötigt GUI-Bibliotheken, die auf Headless-Systemen (z.B. Linux-Server ohne Display) nicht verfügbar sind.

**Lösung**:
```bash
# Deinstallieren Sie opencv-python
pip uninstall opencv-python

# Installieren Sie opencv-python-headless
pip install opencv-python-headless
```

Oder aktualisieren Sie `requirements.txt` und installieren Sie neu:
```bash
pip install -r requirements.txt
```

**Hinweis**: `opencv-python-headless` hat alle Funktionen von `opencv-python`, nur ohne GUI-Abhängigkeiten.

### 2. CUDA Out of Memory

**Fehler**:
```
RuntimeError: CUDA out of memory
```

**Lösungen**:
1. **Batch Size reduzieren**:
   ```yaml
   # In configs/config.yaml
   data:
     batch_size: 2  # Statt 4
   ```

2. **Cache deaktivieren**:
   ```yaml
   data:
     cache_masks: false
   ```

3. **Kleinere Bilder verwenden** (falls möglich)

4. **CPU verwenden** (langsamer):
   ```yaml
   device: "cpu"
   ```

### 3. Windows Multi-Processing Fehler

**Fehler**: `RuntimeError` bei DataLoader mit `num_workers > 0`

**Lösung**: Setzen Sie `num_workers=0` in der Config:
```yaml
data:
  num_workers: 0  # Windows: 0, Linux/Mac: 4-8
```

### 4. ModuleNotFoundError

**Fehler**:
```
ModuleNotFoundError: No module named 'numpy'
```

**Lösung**:
```bash
pip install -r requirements.txt
```

Stellen Sie sicher, dass Sie im richtigen Virtual Environment sind:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 5. TensorBoard nicht verfügbar oder keine Logs sichtbar

**Warnung**: `TensorBoard not available`

**Lösung**:
```bash
pip install tensorboard
```

**Keine Logs in TensorBoard sichtbar?**

1. **TensorBoard im richtigen Verzeichnis starten**:
   ```bash
   # Vom Projekt-Root-Verzeichnis
   tensorboard --logdir outputs/logs
   ```

2. **Warten Sie, bis Logs geschrieben wurden**:
   - Logs werden alle `log_interval` Batches geschrieben (Standard: 10)
   - Prüfen Sie, ob Dateien existieren: `ls outputs/logs/`

3. **Verifizieren Sie die Installation**:
   ```bash
   python scripts/test_tensorboard.py
   ```

4. **Prüfen Sie die Console-Ausgabe beim Training**:
   - Sollte zeigen: "TensorBoard logging enabled. Logs saved to: ..."

Siehe auch: [TensorBoard Anleitung](TENSORBOARD_ANLEITUNG.md)

### 6. Dateien nicht gefunden

**Fehler**: `FileNotFoundError` beim Laden von Daten

**Lösungen**:
1. Überprüfen Sie den Pfad in `configs/config.yaml`:
   ```yaml
   data:
     data_dir: "training_data"  # Muss existieren
   ```

2. Stellen Sie sicher, dass der Ordner `training_data` existiert und die richtige Struktur hat:
   ```
   training_data/
   ├── TCGA-XX-XXXX-01Z-00-DX1/
   │   ├── TCGA-XX-XXXX-01Z-00-DX1_001.tif
   │   ├── TCGA-XX-XXXX-01Z-00-DX1_001.xml
   │   └── ...
   ```

### 7. Checkpoint nicht kompatibel

**Fehler**: Model-Parameter stimmen nicht überein

**Lösung**: Stellen Sie sicher, dass die Model-Konfiguration mit dem trainierten Modell übereinstimmt:
```yaml
model:
  backbone: "resnet34"  # Muss gleich sein wie beim Training!
  decoder_channels: 256  # Muss gleich sein!
  num_types: null  # Muss gleich sein!
```

### 8. Training zu langsam

**Lösungen**:
1. **GPU verwenden**:
   ```yaml
   device: "cuda"
   ```

2. **num_workers erhöhen** (nicht auf Windows):
   ```yaml
   data:
     num_workers: 4  # Oder 8
   ```

3. **Batch Size erhöhen** (wenn GPU-Speicher erlaubt):
   ```yaml
   data:
     batch_size: 8  # Statt 4
   ```

4. **Cache aktivieren** (wenn RAM verfügbar):
   ```yaml
   data:
     cache_masks: true
   ```

### 9. Validation Loss steigt nicht

**Mögliche Ursachen**:
1. Learning Rate zu hoch → Reduzieren Sie auf 1e-5
2. Overfitting → Mehr Augmentation oder Early Stopping
3. Datenproblem → Überprüfen Sie die Annotationen

**Lösung**: Überprüfen Sie die TensorBoard Logs oder Console Output.

### 10. TypeError: got multiple values for keyword argument

**Fehler**:
```
TypeError: AugmentationPipeline() got multiple values for keyword argument 'horizontal_flip'
```

**Ursache**: Die Augmentation-Funktionen hatten ein Problem mit doppelten Argumenten.

**Lösung**: Bereits behoben in der aktuellen Version. Falls das Problem weiterhin auftritt, aktualisieren Sie den Code.

### 11. ValueError: negative strides not supported

**Fehler**:
```
ValueError: At least one stride in the given numpy array is negative, 
and tensors with negative strides are not currently supported.
```

**Ursache**: Nach Augmentation-Operationen (z.B. `flip`) können NumPy-Arrays Views mit negativen Strides haben, die PyTorch nicht unterstützt.

**Lösung**: Bereits behoben in der aktuellen Version. Arrays werden automatisch kopiert, wenn sie nicht kontinuierlich sind.

Falls das Problem weiterhin auftritt, stellen Sie sicher, dass Sie die neueste Version des Codes haben.

### 12. Schlechte Predictions

**Mögliche Ursachen**:
1. Modell nicht ausreichend trainiert → Mehr Epochs
2. Threshold nicht optimal → Threshold anpassen (0.3-0.7)
3. Datenqualität → Überprüfen Sie Ground Truth

**Lösung**:
```python
# Threshold anpassen
inference = HoVerNetInference(
    checkpoint_path='model.pth',
    threshold=0.6  # Experimentieren Sie mit verschiedenen Werten
)
```

## System-spezifische Probleme

### Linux (Headless Server)

- Verwenden Sie `opencv-python-headless`
- Setzen Sie `num_workers` auf 4-8
- GPU: Stellen Sie sicher, dass CUDA installiert ist

### Windows

- Setzen Sie `num_workers=0`
- Verwenden Sie `opencv-python` (GUI ist verfügbar)
- GPU: Installieren Sie PyTorch mit CUDA-Unterstützung

### Mac

- Setzen Sie `num_workers=4`
- GPU: M1/M2 Macs verwenden MPS (Metal Performance Shaders)

## Hilfe bekommen

Wenn das Problem weiterhin besteht:
1. Überprüfen Sie die vollständige Fehlermeldung
2. Überprüfen Sie die Konfigurationsdatei
3. Überprüfen Sie die Dokumentation in `docs/`
4. Überprüfen Sie die Logs in `outputs/logs/`
