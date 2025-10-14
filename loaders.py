"""Dataset loading with parallel processing capabilities."""

import logging
from datetime import datetime
from typing import List, Optional, Iterator
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
from datasets import load_dataset, Dataset

from items import Item
from config import CHUNK_SIZE, MIN_PRICE, MAX_PRICE, DATASET_SOURCE

logger = logging.getLogger(__name__)

class ItemLoader:
    """Loads and processes product datasets from HuggingFace."""

    def __init__(self, name: str):
        self.name = name
        self.dataset: Optional[Dataset] = None

    def from_datapoint(self, datapoint: dict) -> Optional[Item]:
        """Create Item from datapoint if valid, otherwise return None."""
        try:
            price_str = datapoint.get('price')
            if not price_str:
                return None
            
            price = float(price_str)
            if not (MIN_PRICE <= price <= MAX_PRICE):
                return None
            
            item = Item(datapoint, price)
            return item if item.include else None
        
        except (ValueError, TypeError, KeyError):
            return None

    def from_chunk(self, chunk: Dataset) -> List[Item]:
        """Process chunk of datapoints into Items."""
        return [item for datapoint in chunk if (item := self.from_datapoint(datapoint))]

    def chunk_generator(self) -> Iterator[Dataset]:
        """Generate dataset chunks for parallel processing."""
        size = len(self.dataset)
        for i in range(0, size, CHUNK_SIZE):
            yield self.dataset.select(range(i, min(i + CHUNK_SIZE, size)))

    def load_in_parallel(self, workers: int) -> List[Item]:
        """Process dataset chunks in parallel using multiple workers."""
        results = []
        chunk_count = (len(self.dataset) // CHUNK_SIZE) + 1
        
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch in tqdm(
                pool.map(self.from_chunk, self.chunk_generator()),
                total=chunk_count,
                desc=f"Processing {self.name}"
            ):
                results.extend(batch)
        
        for item in results:
            item.category = self.name
        
        return results
            
    def load(self, workers: int = 8) -> List[Item]:
        """Load and process dataset with specified number of workers."""
        start = datetime.now()
        
        logger.info(f"Loading dataset: {self.name}")
        self.dataset = load_dataset(
            DATASET_SOURCE,
            f"raw_meta_{self.name}",
            split="full",
            trust_remote_code=True
        )
        
        results = self.load_in_parallel(workers)
        
        elapsed = (datetime.now() - start).total_seconds() / 60
        logger.info(
            f"Completed {self.name}: {len(results):,} items in {elapsed:.1f} mins"
        )
        
        return results
        

    
    