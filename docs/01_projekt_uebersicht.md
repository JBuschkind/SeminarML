# Projekt-Übersicht

## Was ist HoVer-Net?

HoVer-Net (Horizontal-Vertical Network) ist ein Deep-Learning-Modell zur **Instanz-Segmentierung von Zellkernen** in Histopathologie-Bildern. Es wurde entwickelt, um einzelne Zellen in mikroskopischen Gewebebildern zu erkennen und zu segmentieren.

## Problemstellung

In der Histopathologie müssen Pathologen:
- Zellkerne in Gewebeproben identifizieren
- Verschiedene Zelltypen unterscheiden
- Zellzahlen und -verteilungen analysieren

Dies ist zeitaufwändig und subjektiv. HoVer-Net automatisiert diesen Prozess.

## Was macht HoVer-Net anders?

### Traditionelle Ansätze
- **Semantic Segmentation**: Erkennt Zellen, kann aber nicht zwischen einzelnen Instanzen unterscheiden
- **Instance Segmentation**: Benötigt komplexe Post-Processing (z.B. Watershed)

### HoVer-Net Ansatz
- **Multi-Task Learning**: Lernt gleichzeitig:
  1. **Nuclear Segmentation**: Wo sind Zellkerne?
  2. **HoVer Maps**: Wie können überlappende Zellen getrennt werden?
  3. **Type Classification**: Welcher Zelltyp ist es? (optional)

## Projekt-Struktur

```
diy_impl/
├── training_data/  # Datensatz
├── src/                              # Quellcode
│   ├── data/                         # Datenverarbeitung
│   ├── models/                       # Modell-Architekturen
│   ├── training/                     # Training-Logik
│   ├── evaluation/                   # Evaluation & Visualisierung
│   └── utils/                        # Utilities
├── scripts/                          # Ausführbare Scripts
├── configs/                          # Konfigurationsdateien
├── outputs/                           # Ausgaben (Checkpoints, Logs)
├── data/                              # Verarbeitete Daten
└── docs/                              # Diese Dokumentation
```

## Datenfluss

```
Histopathologie-Bilder (TIF)
    ↓
XML-Annotationen (Aperio ImageScope)
    ↓
XML-Parser extrahiert Polygon-Vertices
    ↓
Masken-Generator erstellt:
  - Nuclear Segmentation Maps
  - Instance Maps
  - HoVer Maps
    ↓
PyTorch Dataset lädt Bilder + Masken
    ↓
DataLoader erstellt Batches
    ↓
HoVer-Net Modell
    ↓
Predictions (Segmentierte Zellen)
    ↓
Post-Processing & Visualisierung
```

## Technologie-Stack

- **Deep Learning**: PyTorch
- **Image Processing**: OpenCV, PIL, scikit-image
- **Scientific Computing**: NumPy, SciPy
- **Visualization**: Matplotlib

## Anwendungsfälle

1. **Medizinische Diagnostik**: Automatische Zellzählung in Biopsien
2. **Forschung**: Analyse von Gewebeproben
3. **Pharmazeutische Entwicklung**: Wirkstoff-Tests

## Nächste Schritte

Lesen Sie als nächstes:
- [Architektur-Übersicht](02_architektur_uebersicht.md) - Wie ist das Projekt aufgebaut?
- [Installation und Setup](03_installation_setup.md) - Wie starte ich?
