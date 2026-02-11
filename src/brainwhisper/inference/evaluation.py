"""Evaluation metrics for EEG-to-text predictions"""

import jiwer
from typing import List, Dict


def calculate_wer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Word Error Rate
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        WER score (lower is better)
    """
    return jiwer.wer(references, predictions)


def calculate_cer(predictions: List[str], references: List[str]) -> float:
    """
    Calculate Character Error Rate
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        CER score (lower is better)
    """
    return jiwer.cer(references, predictions)


def evaluate_results(results: List[Dict]) -> Dict:
    """
    Evaluate prediction results
    
    Args:
        results: List of result dictionaries from batch_predict
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Filter out errors and missing ground truth
    valid_results = [
        r for r in results 
        if r.get('prediction') and r.get('ground_truth') and not r.get('error')
    ]
    
    if not valid_results:
        return {
            'wer': None,
            'cer': None,
            'num_samples': 0,
            'num_errors': len(results)
        }
    
    predictions = [r['prediction'] for r in valid_results]
    references = [r['ground_truth'] for r in valid_results]
    
    wer = calculate_wer(predictions, references)
    cer = calculate_cer(predictions, references)
    
    return {
        'wer': wer,
        'cer': cer,
        'num_samples': len(valid_results),
        'num_errors': len(results) - len(valid_results)
    }


def print_evaluation_report(results: List[Dict]):
    """
    Print detailed evaluation report
    
    Args:
        results: List of result dictionaries from batch_predict
    """
    metrics = evaluate_results(results)
    
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Total samples: {len(results)}")
    print(f"Valid predictions: {metrics['num_samples']}")
    print(f"Errors: {metrics['num_errors']}")
    
    if metrics['wer'] is not None:
        print(f"\nWord Error Rate (WER): {metrics['wer']:.4f}")
        print(f"Character Error Rate (CER): {metrics['cer']:.4f}")
    else:
        print("\nNo valid predictions to evaluate")
    
    print("=" * 60)
