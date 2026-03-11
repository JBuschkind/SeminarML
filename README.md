# Allgemeines

Dieses Repository beinhaltet die Implementierung für unsere Lösung für das Segmentierungsproblem im Rahmen des Seminars "Maschinelles Lernen" im Wintersemester 2025/26.

**Autoren**: Johann Busch, Victoria Gräf, David Kulcitki, Christopher Groß

**Aufbau der Readme**:
- Architektur
- Installation und Verwendung
- Preprocessing

# Architektur

Unsere Architektur basiert auf dem **[HoVer-Net](https://www.sciencedirect.com/science/article/abs/pii/S1361841519301045)** von S. Graham et al. 

## 1. Input Layer
- Verwendung von im Vorfeld augmentierten Daten (npy. Format)
## 2. HoVer-Net Modell
- Encoder (ResNet):  Bildinputs werden in Featuremaps umwandelt
- 3 Branches
    - Nuclear Branch (Wahrscheinlichkeit Pixel gehört zu einem Kern)
    - 2 Hover Branches (2D Vektorfeld für Kernzugehörigkeit)
- Decoder
## 3. Training
- Split Test/Validation Data
- Ablauf:
    - Forward Pass
    - Loss berechnen (siehe Loss Functions)
    - Backward Pass
    - Parameter optimieren

## 4. Evaluation und Visualisierung
- Parameter (Dice Score, Aggregated Jacard Index (AJI), Panoptic Quality (PQ))
- Ausgabe von Bild mit blauumrandeten Zellen

## Loss-Funktionen:
Wir verwenden unterschiedliche Loss Funktionen für die verschiedenen Branches:

### 1. Nuclear Segmentation Loss
* Combined Binary Cross-Entropy (BCE) + Dice Loss
--> binäre Klassifikation (Kern und Hintergrund)

### 2. Hover Map Loss
* L1 Loss (oder L2 Loss)
--> Regression

# Installation und Verwendung

### Voraussetzung:
- Python (getestet mit Version 3.11 und 3.13.11)

### 1. Installation der benötigten Libraries

```bash
pip install -r requirements.txt
```

### 2. Pfad zu Inputbildern hinterlegen
In der Datei /configs/config.yaml in Zeile 34 (data_dir) den Pfad zu den Inputbildern hinterlegen

**Hinweis**: Die Inputs müssen vom Dateityp .npy sein

### 3. Dataloader testen
```bash
python /scripts/test_dataloader.py
```

Dies erstellt:
- Train/Val/Test Split
- Testet den DataLoader
- Generiert Beispiel-Visualisierungen

### 4. Modell trainieren

*Vor dem Training gegebenenfalls die Lernparameter in /configs/config.yaml konfigurieren.*


**Training starten:**
```bash
python scripts/train.py --config configs/config.yaml
```

Das Ergebnis des Trainings sind zwei trainierte Modelle welche in /output/checkpoints gespeichert werden:
- das **beste Modell** (best_model) aus der Testepoche mit dem niedrigsten Fehler
- das **letzte Modell** (last_model) aus der letzten Testepoche

### 5. Testen des Modells
- Zunächst das Data Directory auf den Pfad der Testdaten stellen (siehe 2.)
- Das beste Modell aus dem Training mit den Testdaten evaluieren
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth
```
*alternativ ```--checkpoint outputs/checkpoints/last_model.pth*``` zum Evaluieren des letzten Modells*

Am Ende der Evaluation werden die Testmetriken (Dice Score, F1, Panoptic Quality uvm.) in der Konsole ausgegeben.

Sollen zusätzlich die Testergebnisse der Evaluation als Bilder gespeichert werden, kann noch der Startparameter ```--save-predictions``` hinzugefügt werden:
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pth --save-predictions
```
Die Bilder zeigen für jedes Inputbild einen Vergleich von Orginalbild, Ground Truth-Maske und Prediction-Maske und werden jeweils in /outputs/checkpoints abgespeichert.

### *Häufige Probleme und entsprechende Lösungen sind in der troubleshooting Markdown-Datei beschrieben.*

# Preprocessing
### 1  XML ➔ PNG (Masken-Generierung)
* Logik: Liest Zelltypen (Epithel, Lymphozyt, etc.) aus den XML-Attributen.

* Mapping: Hintergrund = 0, Epithel = 1, Lymphozyt = 2, Neutrophil = 3, Makrophage = 4.

* Technik: Nutzt cv2.fillPoly für pixelgenaue Masken.
### 2 Strategisches Patching (256x256)
* Safe-Zone: Patch-Zentren werden so gewählt, dass keine schwarzen Ränder entstehen (Mindestabstand 128px zum Rand).

* Skalierung:

    < 256px: Zentriertes Padding.

    \> 500px: 3 zufällige Patches.

    \> 900px: 5 zufällige Patches.
### 3 Data Augmentation
* Vervielfältigung des Trainings-Sets (Faktor 3).

* Methoden: Horizontale Spiegelung & zufällige 90°/270° Rotation.

* Synchronität: Alle Transformationen werden identisch auf Bild und Maske angewendet.

### 4 HoVer-Net Export (.npy)
* Format: Einzelne .npy Dateien pro Patch.

* Inhalt: Dictionary mit img (RGB), inst_map (Zelltrennung) und type_map (Zellklassen).
