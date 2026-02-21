# TensorBoard Anleitung

## Installation

TensorBoard sollte bereits in `requirements.txt` enthalten sein. Falls nicht installiert:

```bash
pip install tensorboard
```

## Verifizierung

Testen Sie, ob TensorBoard funktioniert:

```bash
python scripts/test_tensorboard.py
```

Dies sollte:
1. Prüfen, ob TensorBoard installiert ist
2. Test-Logs schreiben
3. Anzeigen, wie TensorBoard gestartet wird

## TensorBoard starten

### Während des Trainings

**Wichtig**: TensorBoard muss im **Projekt-Root-Verzeichnis** gestartet werden!

```bash
# Im Projekt-Root-Verzeichnis
cd ~/SeminarML/diy_impl

# TensorBoard starten
tensorboard --logdir outputs/logs
```

Dann öffnen Sie im Browser: **http://localhost:6006**

### Nach dem Training

Falls das Training bereits läuft oder beendet ist:

```bash
# Im Projekt-Root-Verzeichnis
tensorboard --logdir outputs/logs
```

## Häufige Probleme

### 1. "No dashboards are active"

**Ursache**: TensorBoard wurde im falschen Verzeichnis gestartet oder Logs existieren noch nicht.

**Lösung**:
1. Stellen Sie sicher, dass Sie im Projekt-Root sind
2. Prüfen Sie, ob Logs existieren:
   ```bash
   ls -la outputs/logs/
   ```
3. Warten Sie, bis das Training einige Batches verarbeitet hat (Logs werden alle N Batches geschrieben)

### 2. "Address already in use"

**Ursache**: TensorBoard läuft bereits auf Port 6006.

**Lösung**:
```bash
# Anderen Port verwenden
tensorboard --logdir outputs/logs --port 6007
```

Oder den laufenden Prozess beenden:
```bash
# Port finden
lsof -i :6006

# Prozess beenden
kill <PID>
```

### 3. Keine Logs sichtbar

**Mögliche Ursachen**:

1. **Training hat noch nicht genug Batches verarbeitet**
   - Logs werden alle `log_interval` Batches geschrieben (Standard: 10)
   - Warten Sie einige Minuten

2. **TensorBoard wurde im falschen Verzeichnis gestartet**
   ```bash
   # FALSCH (wenn Sie in einem Unterverzeichnis sind)
   tensorboard --logdir logs
   
   # RICHTIG (vom Projekt-Root)
   tensorboard --logdir outputs/logs
   ```

3. **Logs werden nicht geschrieben**
   - Prüfen Sie, ob TensorBoard verfügbar ist:
     ```python
     python -c "from torch.utils.tensorboard import SummaryWriter; print('OK')"
     ```
   - Prüfen Sie die Console-Ausgabe beim Training:
     - Sollte KEINE Warnung "TensorBoard not available" zeigen

## Was wird geloggt?

### Training Metriken (alle N Batches)
- `Train/BatchLoss` - Loss pro Batch
- `Train/LearningRate` - Aktuelle Learning Rate
- `Train/NuclearLoss` - Nuclear Segmentation Loss
- `Train/HoverLoss` - HoVer Map Loss

### Epoch Metriken (jedes Epoch)
- `Train/EpochLoss` - Durchschnittlicher Loss pro Epoch
- `Train/EpochNuclearLoss` - Durchschnittlicher Nuclear Loss
- `Train/EpochHoverLoss` - Durchschnittlicher HoVer Loss

### Validation Metriken (jedes Epoch)
- `Val/TotalLoss` - Validation Loss
- `Val/NuclearLoss` - Validation Nuclear Loss
- `Val/HoverLoss` - Validation HoVer Loss

## Tipps

1. **Während des Trainings**: TensorBoard kann parallel laufen und aktualisiert sich automatisch
2. **Refresh**: TensorBoard aktualisiert sich automatisch (alle 30 Sekunden)
3. **Mehrere Runs**: TensorBoard zeigt alle Runs im `outputs/logs/` Verzeichnis
4. **Vergleich**: Sie können mehrere Training-Runs vergleichen

## Beispiel-Workflow

```bash
# Terminal 1: Training starten
python scripts/train.py --config configs/config.yaml

# Terminal 2: TensorBoard starten (parallel)
tensorboard --logdir outputs/logs

# Browser: http://localhost:6006 öffnen
```

## Remote Server

Falls Sie auf einem Remote-Server trainieren:

```bash
# Auf dem Server
tensorboard --logdir outputs/logs --host 0.0.0.0 --port 6006

# Auf Ihrem lokalen Rechner (SSH Tunnel)
ssh -L 6006:localhost:6006 user@server

# Dann öffnen Sie im Browser: http://localhost:6006
```

## Verifizierung, dass Logs geschrieben werden

```bash
# Prüfen Sie, ob Dateien erstellt werden
ls -la outputs/logs/

# Sollte zeigen:
# events.out.tfevents.XXXXX
```

Falls keine Dateien vorhanden sind:
1. Warten Sie, bis das Training einige Batches verarbeitet hat
2. Prüfen Sie, ob TensorBoard installiert ist
3. Prüfen Sie die Console-Ausgabe beim Training
