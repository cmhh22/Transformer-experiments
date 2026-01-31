"""
RAG system evaluation module
Implements RAGAS metrics and other evaluations
"""

from typing import List, Dict, Optional
import numpy as np


class RAGEvaluator:
    """RAG system evaluator with multiple metrics"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.metrics = {
            "context_relevance": self._context_relevance,
            "answer_relevance": self._answer_relevance,
            "faithfulness": self._faithfulness,
            "answer_correctness": self._answer_correctness
        }
    
    def evaluate(
        self,
        query: str,
        contexts: List[str],
        answer: str,
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate complete RAG response
        
        Args:
            query: Original question
            contexts: Retrieved contexts
            answer: Generated answer
            ground_truth: Correct answer (optional)
            
        Returns:
            Dict with different metric scores
        """
        results = {}
        
        # Context Relevance: Are the contexts relevant to the query?
        results["context_relevance"] = self._context_relevance(query, contexts)
        
        # Answer Relevance: Is the answer relevant to the query?
        results["answer_relevance"] = self._answer_relevance(query, answer)
        
        # Faithfulness: Is the answer faithful to the contexts?
        results["faithfulness"] = self._faithfulness(answer, contexts)
        
        # Answer Correctness (if ground truth available)
        if ground_truth:
            results["answer_correctness"] = self._answer_correctness(
                answer, ground_truth
            )
        
        return results
    
    def _context_relevance(self, query: str, contexts: List[str]) -> float:
        """
        Evaluate retrieved context relevance
        
        This is a simplified implementation.
        In production, use embeddings or evaluation models.
        """
        if not contexts:
            return 0.0
        
        # Search for query keywords in contexts
        query_terms = set(query.lower().split())
        
        relevance_scores = []
        for context in contexts:
            context_terms = set(context.lower().split())
            overlap = len(query_terms & context_terms)
            score = overlap / len(query_terms) if query_terms else 0
            relevance_scores.append(score)
        
        return np.mean(relevance_scores)
    
    def _answer_relevance(self, query: str, answer: str) -> float:
        """
        Evaluate if the answer is relevant to the question
        """
        if not answer:
            return 0.0
        
        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())
        
        overlap = len(query_terms & answer_terms)
        score = overlap / len(query_terms) if query_terms else 0
        
        return min(score, 1.0)
    
    def _faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Evaluate if the answer is faithful to the contexts
        (doesn't make up information)
        """
        if not answer or not contexts:
            return 0.0
        
        # Check how many statements from the answer
        # are supported by the contexts
        answer_terms = set(answer.lower().split())
        context_text = " ".join(contexts).lower()
        context_terms = set(context_text.split())
        
        supported = len(answer_terms & context_terms)
        total = len(answer_terms)
        
        return supported / total if total > 0 else 0.0
    
    def _answer_correctness(self, answer: str, ground_truth: str) -> float:
        """
        Evaluate answer correctness vs ground truth
        """
        if not answer or not ground_truth:
            return 0.0
        
        # Simple term-based similarity
        answer_terms = set(answer.lower().split())
        gt_terms = set(ground_truth.lower().split())
        
        overlap = len(answer_terms & gt_terms)
        union = len(answer_terms | gt_terms)
        
        # Jaccard similarity
        return overlap / union if union > 0 else 0.0
    
    def evaluate_batch(
        self,
        queries: List[str],
        contexts_list: List[List[str]],
        answers: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """
        Evaluate multiple examples
        
        Args:
            queries: List of queries
            contexts_list: List of context lists
            answers: List of answers
            ground_truths: List of correct answers (optional)
            
        Returns:
            Dict with lists of scores per metric
        """
        results = {metric: [] for metric in self.metrics.keys()}
        
        for i, (query, contexts, answer) in enumerate(
            zip(queries, contexts_list, answers)
        ):
            gt = ground_truths[i] if ground_truths else None
            eval_result = self.evaluate(query, contexts, answer, gt)
            
            for metric, score in eval_result.items():
                results[metric].append(score)
        
        return results
    
    def compute_aggregate_metrics(
        self,
        batch_results: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate aggregate metrics (mean, std, min, max)
        
        Args:
            batch_results: Results from evaluate_batch
            
        Returns:
            Dict with statistics per metric
        """
        aggregated = {}
        
        for metric, scores in batch_results.items():
            if scores:
                aggregated[metric] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "median": np.median(scores)
                }
        
        return aggregated


def print_evaluation_report(results: Dict[str, float]) -> None:
    """
    Print formatted evaluation report
    
    Args:
        results: Dict of metrics
    """
    print("\n" + "=" * 60)
    print("RAG EVALUATION REPORT")
    print("=" * 60)
    
    for metric, score in results.items():
        metric_name = metric.replace("_", " ").title()
        bar = "█" * int(score * 20)
        print(f"{metric_name:20s}: {score:.3f} {bar}")
    
    print("=" * 60)
