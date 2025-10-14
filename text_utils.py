"""Text processing utilities following Single Responsibility Principle."""

import re
from typing import Optional, List
from config import REMOVAL_PATTERNS


class TextCleaner:
    """Handles text cleaning operations."""
    
    @staticmethod
    def clean_details(details: str, patterns: List[str] = REMOVAL_PATTERNS) -> str:
        """Remove common unnecessary patterns from product details."""
        cleaned = details
        for pattern in patterns:
            cleaned = cleaned.replace(pattern, "")
        return cleaned
    
    @staticmethod
    def clean_general_text(text: str) -> str:
        """
        Clean text by removing special characters and filtering product codes.
        Removes words with 7+ characters containing numbers (likely product IDs).
        """
        text = re.sub(r'[:\[\]"{}【】\s]+', ' ', text).strip()
        text = text.replace(" ,", ",").replace(",,,", ",").replace(",,", ",")
        
        words = text.split(' ')
        filtered = [
            word for word in words 
            if len(word) < 7 or not any(char.isdigit() for char in word)
        ]
        return " ".join(filtered)


class ContentAggregator:
    """Aggregates product information from multiple fields."""
    
    @staticmethod
    def combine_fields(
        title: str,
        description: List[str],
        features: List[str],
        details: Optional[str]
    ) -> str:
        """Combine all product fields into single text."""
        parts = []
        
        if description:
            parts.append('\n'.join(description))
        
        if features:
            parts.append('\n'.join(features))
        
        if details:
            cleaned_details = TextCleaner.clean_details(details)
            parts.append(cleaned_details)
        
        return '\n'.join(parts)
