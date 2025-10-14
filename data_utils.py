"""Data processing and analysis utilities."""

import random
import pickle
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

import numpy as np
from datasets import Dataset, DatasetDict

from items import Item
from config import (
    RANDOM_SEED, TRAIN_SIZE, TEST_SIZE,
    PRICE_SLOT_THRESHOLD, MAX_ITEMS_PER_SLOT,
    AUTOMOTIVE_WEIGHT, OTHER_CATEGORY_WEIGHT
)


class DatasetBalancer:
    """Balances dataset by price slots to reduce skew towards cheap items."""
    
    def __init__(
        self,
        slot_threshold: int = PRICE_SLOT_THRESHOLD,
        max_per_slot: int = MAX_ITEMS_PER_SLOT,
        automotive_weight: float = AUTOMOTIVE_WEIGHT,
        other_weight: float = OTHER_CATEGORY_WEIGHT,
        seed: int = RANDOM_SEED
    ):
        self.slot_threshold = slot_threshold
        self.max_per_slot = max_per_slot
        self.automotive_weight = automotive_weight
        self.other_weight = other_weight
        self.seed = seed
    
    def _create_slots(self, items: List[Item]) -> Dict[int, List[Item]]:
        """Group items by rounded price."""
        slots = defaultdict(list)
        for item in items:
            slots[round(item.price)].append(item)
        return slots
    
    def _sample_slot(self, slot_items: List[Item], price: int) -> List[Item]:
        """Sample items from a price slot with category weighting."""
        if price >= self.slot_threshold or len(slot_items) <= self.max_per_slot:
            return slot_items
        
        weights = np.array([
            self.automotive_weight if item.category == 'Automotive' 
            else self.other_weight
            for item in slot_items
        ])
        weights = weights / np.sum(weights)
        
        selected_indices = np.random.choice(
            len(slot_items),
            size=self.max_per_slot,
            replace=False,
            p=weights
        )
        
        return [slot_items[i] for i in selected_indices]
    
    def balance(self, items: List[Item]) -> List[Item]:
        """Balance dataset by sampling from price slots."""
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        slots = self._create_slots(items)
        balanced = []
        
        for price in range(1, 1000):
            if price in slots:
                balanced.extend(self._sample_slot(slots[price], price))
        
        return balanced


class DatasetSplitter:
    """Splits dataset into train and test sets."""
    
    def __init__(
        self,
        train_size: int = TRAIN_SIZE,
        test_size: int = TEST_SIZE,
        seed: int = RANDOM_SEED
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.seed = seed
    
    def split(self, items: List[Item]) -> Tuple[List[Item], List[Item]]:
        """Randomly split items into train and test sets."""
        random.seed(self.seed)
        random.shuffle(items)
        
        train = items[:self.train_size]
        test = items[self.train_size:self.train_size + self.test_size]
        
        return train, test


class DatasetExporter:
    """Exports datasets to various formats."""
    
    @staticmethod
    def to_huggingface(
        train_items: List[Item],
        test_items: List[Item]
    ) -> DatasetDict:
        """Convert items to HuggingFace DatasetDict."""
        train_prompts = [item.prompt for item in train_items]
        train_prices = [item.price for item in train_items]
        
        test_prompts = [item.test_prompt() for item in test_items]
        test_prices = [item.price for item in test_items]
        
        train_dataset = Dataset.from_dict({
            "text": train_prompts,
            "price": train_prices
        })
        
        test_dataset = Dataset.from_dict({
            "text": test_prompts,
            "price": test_prices
        })
        
        return DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })
    
    @staticmethod
    def to_pickle(items: List[Item], filepath: str) -> None:
        """Save items to pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump(items, f)


class DatasetAnalyzer:
    """Analyzes dataset statistics."""
    
    @staticmethod
    def get_statistics(items: List[Item]) -> Dict:
        """Calculate dataset statistics."""
        prices = [item.price for item in items]
        tokens = [item.token_count for item in items]
        categories = Counter(item.category for item in items)
        
        return {
            'count': len(items),
            'price_avg': np.mean(prices),
            'price_max': np.max(prices),
            'price_min': np.min(prices),
            'token_avg': np.mean(tokens),
            'token_max': np.max(tokens),
            'categories': dict(categories)
        }
    
    @staticmethod
    def print_statistics(items: List[Item], name: str = "Dataset") -> None:
        """Print formatted statistics."""
        stats = DatasetAnalyzer.get_statistics(items)
        
        print(f"\n{'='*60}")
        print(f"{name} Statistics")
        print(f"{'='*60}")
        print(f"Total Items: {stats['count']:,}")
        print(f"Price Range: ${stats['price_min']:.2f} - ${stats['price_max']:.2f}")
        print(f"Average Price: ${stats['price_avg']:.2f}")
        print(f"Average Tokens: {stats['token_avg']:.1f}")
        print(f"\nCategory Distribution:")
        for category, count in sorted(stats['categories'].items()):
            print(f"  {category}: {count:,}")
        print(f"{'='*60}\n")
