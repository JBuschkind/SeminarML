"""
XML Parser for Aperio ImageScope Annotation Format
Parses XML files and extracts polygon vertices for each cell/nucleus
"""

import xml.etree.ElementTree as ET
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path


def parse_xml_annotations(xml_path: str) -> Dict:
    """
    Parse Aperio ImageScope XML annotation file.
    
    Args:
        xml_path: Path to XML annotation file
        
    Returns:
        Dictionary containing:
            - 'regions': List of regions, each with 'vertices' (Nx2 array) and 'id'
            - 'microns_per_pixel': Resolution information
            - 'num_regions': Number of annotated regions
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extract microns per pixel
    microns_per_pixel = float(root.attrib.get('MicronsPerPixel', 0.0))
    
    regions = []
    region_id = 0
    
    # Iterate through all annotations
    for annotation in root.findall('.//Annotation'):
        # Find all regions in this annotation
        regions_elem = annotation.find('Regions')
        if regions_elem is not None:
            for region in regions_elem.findall('Region'):
                region_id += 1
                vertices = []
                
                # Extract vertices
                vertices_elem = region.find('Vertices')
                if vertices_elem is not None:
                    for vertex in vertices_elem.findall('Vertex'):
                        x = float(vertex.attrib['X'])
                        y = float(vertex.attrib['Y'])
                        vertices.append([x, y])
                
                if len(vertices) > 0:
                    # Remove duplicate consecutive vertices
                    vertices_array = np.array(vertices)
                    # Keep only unique consecutive vertices
                    unique_mask = np.ones(len(vertices_array), dtype=bool)
                    for i in range(1, len(vertices_array)):
                        if np.allclose(vertices_array[i], vertices_array[i-1], atol=1e-6):
                            unique_mask[i] = False
                    vertices_array = vertices_array[unique_mask]
                    
                    # Ensure polygon is closed (first == last)
                    if len(vertices_array) > 0:
                        if not np.allclose(vertices_array[0], vertices_array[-1], atol=1e-6):
                            vertices_array = np.vstack([vertices_array, vertices_array[0:1]])
                    
                    regions.append({
                        'id': region_id,
                        'vertices': vertices_array,
                        'area': float(region.attrib.get('Area', 0.0)),
                        'length': float(region.attrib.get('Length', 0.0))
                    })
    
    return {
        'regions': regions,
        'microns_per_pixel': microns_per_pixel,
        'num_regions': len(regions)
    }


def get_region_bounds(regions: List[Dict]) -> Tuple[int, int, int, int]:
    """
    Get bounding box of all regions.
    
    Args:
        regions: List of region dictionaries
        
    Returns:
        (min_x, min_y, max_x, max_y)
    """
    if not regions:
        return 0, 0, 0, 0
    
    all_vertices = np.vstack([r['vertices'] for r in regions])
    min_x, min_y = np.floor(all_vertices.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_vertices.max(axis=0)).astype(int)
    
    return min_x, min_y, max_x, max_y


if __name__ == '__main__':
    # Test the parser
    import sys
    if len(sys.argv) > 1:
        xml_path = sys.argv[1]
        result = parse_xml_annotations(xml_path)
        print(f"Parsed {result['num_regions']} regions")
        print(f"Microns per pixel: {result['microns_per_pixel']}")
        if result['regions']:
            print(f"First region has {len(result['regions'][0]['vertices'])} vertices")
