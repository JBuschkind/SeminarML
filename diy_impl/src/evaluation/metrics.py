"""
Evaluation Metrics for HoVer-Net
Implements metrics for nucleus instance segmentation:
- Dice Score
- Aggregated Jaccard Index (AJI)
- Panoptic Quality (PQ)
- Pixel Accuracy
- F1 Score
"""

import torch
import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage.segmentation import watershed
from typing import Dict, Tuple, Optional
import cv2


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """
    Compute Dice Score for binary segmentation.
    
    Args:
        pred: Binary predictions (H, W) in {0, 1}
        target: Binary ground truth (H, W) in {0, 1}
        smooth: Smoothing factor
        
    Returns:
        Dice score (0-1, higher is better)
    """
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return float(dice)


def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute pixel accuracy.
    
    Args:
        pred: Predictions (H, W)
        target: Ground truth (H, W)
        
    Returns:
        Pixel accuracy (0-1, higher is better)
    """
    correct = (pred == target).sum()
    total = pred.size
    return float(correct / total) if total > 0 else 0.0


def get_instance_map_from_predictions(
    nuclear_pred: np.ndarray,
    hover_pred: np.ndarray,
    threshold: float = 0.5,
    min_distance: int = 10,
    min_instance_size: int = 30,
) -> np.ndarray:
    """
    Convert nuclear and hover predictions to instance map using watershed.
    
    Args:
        nuclear_pred: Nuclear segmentation prediction (H, W) in [0, 1]
        hover_pred: HoVer map prediction (H, W, 2) in [-1, 1]
        threshold: Threshold for nuclear segmentation
        min_distance: Minimum pixel distance between markers (larger = fewer instances)
        min_instance_size: Remove instances smaller than this many pixels (reduces over-segmentation)
        
    Returns:
        Instance map (H, W) with unique IDs
    """
    # Binary nuclear mask
    nuclear_binary = (nuclear_pred > threshold).astype(np.uint8)
    
    if nuclear_binary.sum() == 0:
        return np.zeros_like(nuclear_binary, dtype=np.int32)
    
    # Compute distance transform for markers
    dist_transform = ndimage.distance_transform_edt(nuclear_binary)
    
    # Find local maxima as markers (larger min_distance => fewer, more stable markers)
    marker_threshold = 0.3 * dist_transform.max()
    from scipy.ndimage import maximum_filter
    local_maxima = maximum_filter(dist_transform, size=min_distance) == dist_transform
    local_maxima = local_maxima & (dist_transform > marker_threshold)
    
    # Create markers
    markers = np.zeros_like(nuclear_binary, dtype=np.int32)
    marker_coords = np.where(local_maxima)
    
    if len(marker_coords[0]) > 0:
        for idx, (y, x) in enumerate(zip(marker_coords[0], marker_coords[1])):
            markers[y, x] = idx + 1
    
    # If no markers found, use distance transform directly
    if markers.sum() == 0:
        markers = (dist_transform > 0.5 * dist_transform.max()).astype(np.int32)
        markers = ndimage.label(markers)[0]
    
    # Create watershed mask using HoVer maps
    hover_magnitude = np.sqrt(hover_pred[:, :, 0]**2 + hover_pred[:, :, 1]**2)
    elevation = 1.0 - np.clip(hover_magnitude, 0, 1)
    
    # Apply watershed
    labels = watershed(elevation, markers, mask=nuclear_binary)
    labels = labels.astype(np.int32)
    
    # Remove tiny instances (over-segmentation) and relabel
    if min_instance_size > 0:
        unique_ids = np.unique(labels)
        unique_ids = unique_ids[unique_ids > 0]
        new_labels = np.zeros_like(labels)
        next_id = 1
        for uid in unique_ids:
            if (labels == uid).sum() >= min_instance_size:
                new_labels[labels == uid] = next_id
                next_id += 1
        labels = new_labels
    
    return labels


def compute_iou(inst1: np.ndarray, inst2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two instances.
    
    Args:
        inst1: Binary mask of instance 1 (H, W)
        inst2: Binary mask of instance 2 (H, W)
        
    Returns:
        IoU score (0-1)
    """
    intersection = (inst1 & inst2).sum()
    union = (inst1 | inst2).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection / union)


def aggregated_jaccard_index(
    pred_instances: np.ndarray,
    target_instances: np.ndarray,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute Aggregated Jaccard Index (AJI).
    
    AJI measures instance segmentation quality by matching predicted and ground truth instances.
    
    Args:
        pred_instances: Predicted instance map (H, W) with unique IDs
        target_instances: Ground truth instance map (H, W) with unique IDs
        iou_threshold: IoU threshold for matching instances
        
    Returns:
        Dictionary with:
            - 'aji': AJI score (0-1, higher is better)
            - 'precision': Precision
            - 'recall': Recall
            - 'f1': F1 score
    """
    # Get unique instance IDs (excluding background = 0)
    pred_ids = np.unique(pred_instances)
    pred_ids = pred_ids[pred_ids > 0]
    target_ids = np.unique(target_instances)
    target_ids = target_ids[target_ids > 0]
    
    if len(pred_ids) == 0 and len(target_ids) == 0:
        return {'aji': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    
    if len(pred_ids) == 0:
        return {'aji': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    if len(target_ids) == 0:
        return {'aji': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(target_ids), len(pred_ids)))
    
    for i, target_id in enumerate(target_ids):
        target_mask = (target_instances == target_id)
        for j, pred_id in enumerate(pred_ids):
            pred_mask = (pred_instances == pred_id)
            iou_matrix[i, j] = compute_iou(target_mask, pred_mask)
    
    # Match instances (Hungarian algorithm approximation - greedy matching)
    matched_targets = set()
    matched_predictions = set()
    matches = []
    
    # Sort by IoU (descending)
    match_candidates = []
    for i in range(len(target_ids)):
        for j in range(len(pred_ids)):
            if iou_matrix[i, j] >= iou_threshold:
                match_candidates.append((iou_matrix[i, j], i, j))
    
    match_candidates.sort(reverse=True)
    
    for iou, i, j in match_candidates:
        if i not in matched_targets and j not in matched_predictions:
            matches.append((i, j, iou))
            matched_targets.add(i)
            matched_predictions.add(j)
    
    # Compute AJI
    intersection_sum = 0.0
    union_sum = 0.0
    
    # Matched instances
    for target_idx, pred_idx, iou in matches:
        target_id = target_ids[target_idx]
        pred_id = pred_ids[pred_idx]
        
        target_mask = (target_instances == target_id)
        pred_mask = (pred_instances == pred_id)
        
        intersection = (target_mask & pred_mask).sum()
        union = (target_mask | pred_mask).sum()
        
        intersection_sum += intersection
        union_sum += union
    
    # Unmatched target instances
    for i, target_id in enumerate(target_ids):
        if i not in matched_targets:
            target_mask = (target_instances == target_id)
            union_sum += target_mask.sum()
    
    # Unmatched predicted instances
    for j, pred_id in enumerate(pred_ids):
        if j not in matched_predictions:
            pred_mask = (pred_instances == pred_id)
            union_sum += pred_mask.sum()
    
    # AJI
    if union_sum == 0:
        aji = 0.0
    else:
        aji = intersection_sum / union_sum
    
    # Precision, Recall, F1
    tp = len(matches)
    fp = len(pred_ids) - len(matched_predictions)
    fn = len(target_ids) - len(matched_targets)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'aji': float(aji),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn)
    }


def panoptic_quality(
    pred_instances: np.ndarray,
    target_instances: np.ndarray,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute Panoptic Quality (PQ).
    
    PQ = (IoU of matched instances) * (Detection Quality)
    
    Args:
        pred_instances: Predicted instance map (H, W)
        target_instances: Ground truth instance map (H, W)
        iou_threshold: IoU threshold for matching
        
    Returns:
        Dictionary with:
            - 'pq': Panoptic Quality (0-1, higher is better)
            - 'sq': Segmentation Quality (average IoU of matched instances)
            - 'dq': Detection Quality (precision * recall)
    """
    # Get unique instance IDs
    pred_ids = np.unique(pred_instances)
    pred_ids = pred_ids[pred_ids > 0]
    target_ids = np.unique(target_instances)
    target_ids = target_ids[target_ids > 0]
    
    if len(pred_ids) == 0 and len(target_ids) == 0:
        return {'pq': 1.0, 'sq': 1.0, 'dq': 1.0}
    
    if len(pred_ids) == 0 or len(target_ids) == 0:
        return {'pq': 0.0, 'sq': 0.0, 'dq': 0.0}
    
    # Compute IoU matrix and match instances
    iou_matrix = np.zeros((len(target_ids), len(pred_ids)))
    
    for i, target_id in enumerate(target_ids):
        target_mask = (target_instances == target_id)
        for j, pred_id in enumerate(pred_ids):
            pred_mask = (pred_instances == pred_id)
            iou_matrix[i, j] = compute_iou(target_mask, pred_mask)
    
    # Greedy matching
    matched_targets = set()
    matched_predictions = set()
    matched_ious = []
    
    match_candidates = []
    for i in range(len(target_ids)):
        for j in range(len(pred_ids)):
            if iou_matrix[i, j] >= iou_threshold:
                match_candidates.append((iou_matrix[i, j], i, j))
    
    match_candidates.sort(reverse=True)
    
    for iou, i, j in match_candidates:
        if i not in matched_targets and j not in matched_predictions:
            matched_ious.append(iou)
            matched_targets.add(i)
            matched_predictions.add(j)
    
    # Segmentation Quality (SQ): Average IoU of matched instances
    sq = np.mean(matched_ious) if len(matched_ious) > 0 else 0.0
    
    # Detection Quality (DQ): Precision * Recall
    tp = len(matched_ious)
    fp = len(pred_ids) - len(matched_predictions)
    fn = len(target_ids) - len(matched_targets)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dq = precision * recall
    
    # Panoptic Quality
    pq = sq * dq
    
    return {
        'pq': float(pq),
        'sq': float(sq),
        'dq': float(dq),
        'precision': float(precision),
        'recall': float(recall)
    }


def evaluate_predictions(
    nuclear_pred: np.ndarray,
    hover_pred: np.ndarray,
    nuclear_target: np.ndarray,
    instance_target: np.ndarray,
    threshold: float = 0.5,
    min_distance: int = 10,
    min_instance_size: int = 30,
) -> Dict[str, float]:
    """
    Evaluate predictions against ground truth.
    
    Args:
        nuclear_pred: Nuclear segmentation prediction (H, W) in [0, 1]
        hover_pred: HoVer map prediction (H, W, 2) in [-1, 1]
        nuclear_target: Ground truth nuclear segmentation (H, W) in {0, 1}
        instance_target: Ground truth instance map (H, W) with unique IDs
        threshold: Threshold for nuclear segmentation
        min_distance: Min pixel distance between watershed markers
        min_instance_size: Remove predicted instances smaller than this (pixels)
        
    Returns:
        Dictionary with all metrics
    """
    # Convert predictions to instance map
    pred_instances = get_instance_map_from_predictions(
        nuclear_pred, hover_pred,
        threshold=threshold,
        min_distance=min_distance,
        min_instance_size=min_instance_size,
    )
    
    # Ensure target is numpy array
    if isinstance(nuclear_target, torch.Tensor):
        nuclear_target = nuclear_target.cpu().numpy()
    if isinstance(instance_target, torch.Tensor):
        instance_target = instance_target.cpu().numpy()
    
    # Binary nuclear segmentation metrics
    nuclear_binary = (nuclear_pred > threshold).astype(np.uint8)
    dice = dice_score(nuclear_binary, nuclear_target.astype(np.uint8))
    pixel_acc = pixel_accuracy(nuclear_binary, nuclear_target.astype(np.uint8))
    
    # Instance segmentation metrics
    aji_metrics = aggregated_jaccard_index(pred_instances, instance_target)
    pq_metrics = panoptic_quality(pred_instances, instance_target)
    
    # Combine all metrics
    results = {
        'dice': dice,
        'pixel_accuracy': pixel_acc,
        'aji': aji_metrics['aji'],
        'precision': aji_metrics['precision'],
        'recall': aji_metrics['recall'],
        'f1': aji_metrics['f1'],
        'pq': pq_metrics['pq'],
        'sq': pq_metrics['sq'],
        'dq': pq_metrics['dq'],
        'num_pred_instances': len(np.unique(pred_instances)) - 1,  # Exclude background
        'num_target_instances': len(np.unique(instance_target)) - 1
    }
    
    return results


if __name__ == '__main__':
    # Test metrics
    print("Testing evaluation metrics...")
    
    # Create dummy data
    h, w = 256, 256
    nuclear_pred = np.random.rand(h, w)
    hover_pred = np.random.randn(h, w, 2)
    nuclear_target = np.random.randint(0, 2, (h, w))
    instance_target = np.random.randint(0, 5, (h, w))
    
    # Evaluate
    results = evaluate_predictions(
        nuclear_pred, hover_pred, nuclear_target, instance_target
    )
    
    print("\nEvaluation Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
