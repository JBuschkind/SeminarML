# XML-Parser Dokumentation

## Übersicht

Der XML-Parser (`src/data/xml_parser.py`) konvertiert Aperio ImageScope XML-Annotationen in ein Python-Format, das für die Masken-Generierung verwendet wird.

## Was macht der XML-Parser?

1. **Liest XML-Dateien** im Aperio ImageScope Format
2. **Extrahiert Polygon-Vertices** für jede annotierte Zelle
3. **Bereinigt die Daten** (entfernt doppelte Vertices)
4. **Gibt strukturierte Daten** zurück

## XML-Format (Aperio ImageScope)

### Struktur

```xml
<Annotations MicronsPerPixel="0.252000">
  <Annotation Id="1" ...>
    <Regions>
      <Region Id="1" Area="918.0" ...>
        <Vertices>
          <Vertex X="372" Y="551" Z="0"/>
          <Vertex X="370" Y="551" Z="0"/>
          ...
        </Vertices>
      </Region>
      <Region Id="2" ...>
        ...
      </Region>
    </Regions>
  </Annotation>
</Annotations>
```

### Wichtige Elemente

- **`<Annotations>`**: Root-Element, enthält `MicronsPerPixel` (Auflösung)
- **`<Annotation>`**: Eine Annotation-Gruppe (kann mehrere Regionen enthalten)
- **`<Region>`**: Eine einzelne Zelle/Nukleus
- **`<Vertex>`**: Ein Punkt auf dem Polygon (X, Y Koordinaten)

## API Referenz

### `parse_xml_annotations(xml_path: str) -> Dict`

Parst eine XML-Datei und gibt strukturierte Daten zurück.

**Parameter**:
- `xml_path` (str): Pfad zur XML-Datei

**Rückgabe** (Dict):
```python
{
    'regions': [
        {
            'id': 1,
            'vertices': np.array([[x1, y1], [x2, y2], ...]),  # (N, 2)
            'area': 918.0,
            'length': 134.6
        },
        ...
    ],
    'microns_per_pixel': 0.252,
    'num_regions': 42
}
```

**Beispiel**:
```python
from src.data.xml_parser import parse_xml_annotations

# XML-Datei parsen
annotations = parse_xml_annotations("path/to/annotation.xml")

print(f"Anzahl Regionen: {annotations['num_regions']}")
print(f"Auflösung: {annotations['microns_per_pixel']} μm/pixel")

# Erste Region anzeigen
first_region = annotations['regions'][0]
print(f"Region {first_region['id']} hat {len(first_region['vertices'])} Vertices")
```

## Implementierungs-Details

### 1. XML-Parsing

```python
tree = ET.parse(xml_path)
root = tree.getroot()
```

Verwendet `xml.etree.ElementTree` (Standard-Bibliothek).

### 2. Vertex-Extraktion

```python
for vertex in vertices_elem.findall('Vertex'):
    x = float(vertex.attrib['X'])
    y = float(vertex.attrib['Y'])
    vertices.append([x, y])
```

Koordinaten werden als Float gespeichert (können auch Dezimalzahlen sein).

### 3. Duplikat-Entfernung

```python
# Entferne aufeinanderfolgende Duplikate
unique_mask = np.ones(len(vertices_array), dtype=bool)
for i in range(1, len(vertices_array)):
    if np.allclose(vertices_array[i], vertices_array[i-1], atol=1e-6):
        unique_mask[i] = False
vertices_array = vertices_array[unique_mask]
```

**Warum?** Manchmal haben XML-Dateien doppelte Vertices, die Probleme verursachen können.

### 4. Polygon-Schließung

```python
# Stelle sicher, dass Polygon geschlossen ist
if not np.allclose(vertices_array[0], vertices_array[-1], atol=1e-6):
    vertices_array = np.vstack([vertices_array, vertices_array[0:1]])
```

**Warum?** Polygone müssen geschlossen sein (erster Punkt = letzter Punkt).

## Verwendung im Projekt

Der XML-Parser wird automatisch vom Dataset verwendet:

```python
# In dataset.py
from .xml_parser import parse_xml_annotations

annotations = parse_xml_annotations(annotation_path)
regions = annotations['regions']
# ... weiter zur Masken-Generierung
```

## Fehlerbehandlung

### Häufige Probleme

1. **Datei nicht gefunden**
   ```python
   FileNotFoundError: [Errno 2] No such file or directory
   ```
   **Lösung**: Überprüfen Sie den Pfad

2. **Ungültiges XML-Format**
   ```python
   xml.etree.ElementTree.ParseError
   ```
   **Lösung**: Überprüfen Sie, ob die XML-Datei korrekt formatiert ist

3. **Leere Annotationen**
   - Wenn keine Regionen gefunden werden, gibt `regions` eine leere Liste zurück
   - Das Dataset behandelt dies korrekt (leere Masken)

## Testen

```python
# Test-Script
python -m src.data.xml_parser path/to/annotation.xml
```

## Nächste Schritte

- [Masken-Generierung](05_masken_generierung.md) - Wie werden aus Polygonen Masken?
