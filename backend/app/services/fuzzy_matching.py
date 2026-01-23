"""
Fuzzy Matching Utility Module
Uses RapidFuzz for fast and accurate string matching
"""
from typing import List, Tuple, Optional
from rapidfuzz import fuzz, process
import logging

logger = logging.getLogger(__name__)


class FuzzyMatcher:
    """
    Fuzzy matching utility using RapidFuzz library.
    Provides faster and more accurate matching than SequenceMatcher.
    """

    @staticmethod
    def ratio(target: str, candidate: str, normalize: bool = True) -> float:
        """
        Calculate similarity ratio between two strings using RapidFuzz.

        Args:
            target: Target string to match
            candidate: Candidate string to compare
            normalize: Whether to normalize strings before comparison

        Returns:
            Similarity score between 0.0 and 100.0 (RapidFuzz uses 0-100 scale)
        """
        if normalize:
            target = FuzzyMatcher._normalize_string(target)
            candidate = FuzzyMatcher._normalize_string(candidate)

        return fuzz.ratio(target, candidate)

    @staticmethod
    def partial_ratio(target: str, candidate: str, normalize: bool = True) -> float:
        """
        Calculate partial similarity ratio (best substring match).
        Useful for matching when one string is a substring of another.

        Args:
            target: Target string to match
            candidate: Candidate string to compare
            normalize: Whether to normalize strings before comparison

        Returns:
            Similarity score between 0.0 and 100.0
        """
        if normalize:
            target = FuzzyMatcher._normalize_string(target)
            candidate = FuzzyMatcher._normalize_string(candidate)

        return fuzz.partial_ratio(target, candidate)

    @staticmethod
    def token_sort_ratio(target: str, candidate: str, normalize: bool = True) -> float:
        """
        Calculate similarity after sorting tokens (words).
        Useful for matching strings with same words in different order.

        Args:
            target: Target string to match
            candidate: Candidate string to compare
            normalize: Whether to normalize strings before comparison

        Returns:
            Similarity score between 0.0 and 100.0
        """
        if normalize:
            target = FuzzyMatcher._normalize_string(target)
            candidate = FuzzyMatcher._normalize_string(candidate)

        return fuzz.token_sort_ratio(target, candidate)

    @staticmethod
    def token_set_ratio(target: str, candidate: str, normalize: bool = True) -> float:
        """
        Calculate similarity using token set matching.
        Best for matching strings with overlapping words but different lengths.

        Args:
            target: Target string to match
            candidate: Candidate string to compare
            normalize: Whether to normalize strings before comparison

        Returns:
            Similarity score between 0.0 and 100.0
        """
        if normalize:
            target = FuzzyMatcher._normalize_string(target)
            candidate = FuzzyMatcher._normalize_string(candidate)

        return fuzz.token_set_ratio(target, candidate)

    @staticmethod
    def weighted_ratio(target: str, candidate: str, normalize: bool = True) -> float:
        """
        Calculate weighted similarity ratio using multiple algorithms.
        Combines ratio, partial_ratio, token_sort_ratio, and token_set_ratio
        for the best overall match score.

        Args:
            target: Target string to match
            candidate: Candidate string to compare
            normalize: Whether to normalize strings before comparison

        Returns:
            Similarity score between 0.0 and 100.0
        """
        if normalize:
            target = FuzzyMatcher._normalize_string(target)
            candidate = FuzzyMatcher._normalize_string(candidate)

        return fuzz.WRatio(target, candidate)

    @staticmethod
    def match(
        target: str, candidate: str, threshold: float = 80.0, use_weighted: bool = True
    ) -> Tuple[bool, float]:
        """
        Match a target string against a candidate with a threshold.

        Args:
            target: Target string to match
            candidate: Candidate string to compare
            threshold: Minimum similarity score (0-100 scale)
            use_weighted: Whether to use weighted_ratio (best) or simple ratio

        Returns:
            Tuple of (is_match, similarity_score)
        """
        if use_weighted:
            score = FuzzyMatcher.weighted_ratio(target, candidate)
        else:
            score = FuzzyMatcher.ratio(target, candidate)

        return (score >= threshold, score)

    @staticmethod
    def find_best_matches(
        target: str,
        candidates: List[str],
        limit: int = 10,
        threshold: float = 80.0,
        use_weighted: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Find best matching candidates for a target string.

        Args:
            target: Target string to match
            candidates: List of candidate strings
            limit: Maximum number of results to return
            threshold: Minimum similarity score (0-100 scale)
            use_weighted: Whether to use weighted_ratio (best) or simple ratio

        Returns:
            List of (candidate, similarity_score) tuples, sorted by score (descending)
        """
        if not candidates:
            return []

        # Use RapidFuzz process.extract for efficient batch matching
        scorer = fuzz.WRatio if use_weighted else fuzz.ratio

        results = process.extract(target, candidates, scorer=scorer, limit=limit)

        # Filter by threshold and convert to list of tuples
        matches = [
            (candidate, score) for candidate, score, _ in results if score >= threshold
        ]

        return matches

    @staticmethod
    def find_best_match(
        target: str,
        candidates: List[str],
        threshold: float = 80.0,
        use_weighted: bool = True,
    ) -> Optional[Tuple[str, float]]:
        """
        Find the best matching candidate for a target string.

        Args:
            target: Target string to match
            candidates: List of candidate strings
            threshold: Minimum similarity score (0-100 scale)
            use_weighted: Whether to use weighted_ratio (best) or simple ratio

        Returns:
            Tuple of (best_candidate, similarity_score) or None if no match above threshold
        """
        matches = FuzzyMatcher.find_best_matches(
            target, candidates, limit=1, threshold=threshold, use_weighted=use_weighted
        )

        return matches[0] if matches else None

    @staticmethod
    def _normalize_string(s: str) -> str:
        """
        Normalize string for matching (lowercase, strip whitespace).
        More comprehensive normalization should be done by the caller.

        Args:
            s: String to normalize

        Returns:
            Normalized string
        """
        if not s:
            return ""
        return str(s).lower().strip()
