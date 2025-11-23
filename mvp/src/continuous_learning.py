"""
Automated Continuous Learning Module

This module implements automatic retraining from user corrections.
When corrections are submitted, they are automatically integrated into the index
to improve future predictions without manual intervention.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple

logger = logging.getLogger("txcat.learning")

# Track which corrections have been processed to avoid duplicates
_PROCESSED_CORRECTIONS_FILE = Path(__file__).parent.parent / "data" / "processed_corrections.json"


def load_processed_correction_ids() -> Set[str]:
    """Load set of already processed correction IDs."""
    if not _PROCESSED_CORRECTIONS_FILE.exists():
        return set()
    try:
        with open(_PROCESSED_CORRECTIONS_FILE, "r", encoding="utf8") as fh:
            data = json.load(fh)
            return set(data.get("processed_ids", []))
    except Exception as e:
        logger.warning("Failed to load processed corrections: %s", e)
        return set()


def save_processed_correction_ids(processed_ids: Set[str]):
    """Save set of processed correction IDs."""
    _PROCESSED_CORRECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(_PROCESSED_CORRECTIONS_FILE, "w", encoding="utf8") as fh:
            json.dump({"processed_ids": list(processed_ids)}, fh, indent=2)
    except Exception as e:
        logger.warning("Failed to save processed corrections: %s", e)


def load_corrections_from_buffer(corrections_path: Path) -> List[Dict[str, Any]]:
    """Load corrections from JSONL buffer file."""
    if not corrections_path.exists():
        return []
    
    corrections = []
    try:
        with open(corrections_path, "r", encoding="utf8") as fh:
            for line_num, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    correction = json.loads(line)
                    # Create unique ID from transaction + timestamp
                    correction_id = f"{correction.get('transaction', '')}_{correction.get('ts', 0)}"
                    correction["_id"] = correction_id
                    corrections.append(correction)
                except json.JSONDecodeError as e:
                    logger.warning("Invalid JSON in corrections line %d: %s", line_num, e)
                    continue
    except Exception as e:
        logger.error("Failed to load corrections: %s", e)
    
    return corrections


def extract_new_corrections(
    corrections: List[Dict[str, Any]],
    processed_ids: Set[str]
) -> List[Dict[str, Any]]:
    """Extract corrections that haven't been processed yet."""
    new_corrections = []
    for correction in corrections:
        correction_id = correction.get("_id", "")
        if correction_id and correction_id not in processed_ids:
            new_corrections.append(correction)
    return new_corrections


def auto_retrain_from_corrections(
    indexer,
    corrections_path: Path,
    min_corrections: int = 1,
    max_corrections_per_batch: int = 100
) -> Dict[str, Any]:
    """
    Automatically retrain the index from new corrections.
    
    Args:
        indexer: The TxIndexer instance to update
        corrections_path: Path to corrections_buffer.jsonl
        min_corrections: Minimum number of new corrections before retraining
        max_corrections_per_batch: Maximum corrections to process in one batch
    
    Returns:
        Dictionary with retraining statistics
    """
    logger.info("Starting automatic retraining from corrections...")
    
    # Load corrections
    all_corrections = load_corrections_from_buffer(corrections_path)
    if not all_corrections:
        logger.info("No corrections found")
        return {
            "status": "no_corrections",
            "processed": 0,
            "added_to_index": 0,
            "total_corrections": 0
        }
    
    # Load processed IDs
    processed_ids = load_processed_correction_ids()
    
    # Extract new corrections
    new_corrections = extract_new_corrections(all_corrections, processed_ids)
    
    if len(new_corrections) < min_corrections:
        logger.info("Not enough new corrections (%d < %d)", len(new_corrections), min_corrections)
        return {
            "status": "insufficient_corrections",
            "processed": 0,
            "added_to_index": 0,
            "new_corrections": len(new_corrections),
            "min_required": min_corrections
        }
    
    # Limit batch size
    batch = new_corrections[:max_corrections_per_batch]
    
    # Add corrections to index
    added_count = 0
    skipped_count = 0
    
    for correction in batch:
        transaction = correction.get("transaction", "").strip()
        correct_label = correction.get("correct_label", "").strip()
        
        if not transaction or not correct_label:
            skipped_count += 1
            logger.warning("Skipping invalid correction: %s", correction)
            continue
        
        # Add to index
        try:
            indexer.add_document(
                text=transaction,
                label=correct_label,
                meta={"source": "user_correction", "timestamp": correction.get("ts")},
                rebuild=False  # Rebuild once at the end for efficiency
            )
            added_count += 1
        except Exception as e:
            logger.error("Failed to add correction to index: %s", e)
            skipped_count += 1
    
    # Rebuild index with all new documents
    if added_count > 0:
        logger.info("Rebuilding index with %d new corrections...", added_count)
        try:
            indexer.build_index()
            logger.info("Index rebuilt successfully. Total documents: %d", len(indexer.docs))
        except Exception as e:
            logger.error("Failed to rebuild index: %s", e)
            return {
                "status": "rebuild_failed",
                "processed": len(batch),
                "added_to_index": added_count,
                "error": str(e)
            }
    
    # Mark corrections as processed
    new_processed_ids = {c.get("_id", "") for c in batch if c.get("_id")}
    processed_ids.update(new_processed_ids)
    save_processed_correction_ids(processed_ids)
    
    result = {
        "status": "success",
        "processed": len(batch),
        "added_to_index": added_count,
        "skipped": skipped_count,
        "total_corrections": len(all_corrections),
        "new_corrections": len(new_corrections),
        "index_size": len(indexer.docs)
    }
    
    logger.info("Automatic retraining complete: %s", result)
    return result


def get_retraining_stats(corrections_path: Path) -> Dict[str, Any]:
    """Get statistics about corrections and retraining status."""
    all_corrections = load_corrections_from_buffer(corrections_path)
    processed_ids = load_processed_correction_ids()
    new_corrections = extract_new_corrections(all_corrections, processed_ids)
    
    # Count corrections by category
    category_counts = {}
    for correction in all_corrections:
        label = correction.get("correct_label", "UNKNOWN")
        category_counts[label] = category_counts.get(label, 0) + 1
    
    return {
        "total_corrections": len(all_corrections),
        "processed_corrections": len(processed_ids),
        "new_corrections": len(new_corrections),
        "corrections_by_category": category_counts,
        "ready_for_retraining": len(new_corrections) > 0
    }

