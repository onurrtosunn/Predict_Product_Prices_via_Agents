import os
import json
import pickle
import random
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset, Dataset, DatasetDict
import matplotlib.pyplot as plt


# Abstract Interfaces (Dependency Inversion Principle)
class DataLoaderInterface(ABC):
    """Interface for data loading operations"""
    
    @abstractmethod
    def load_dataset(self, category: str) -> Any:
        pass


class DataCuratorInterface(ABC):
    """Interface for data curation operations"""
    
    @abstractmethod
    def curate_items(self, dataset: Any) -> List[Any]:
        pass


class DataSplitterInterface(ABC):
    """Interface for data splitting operations"""
    
    @abstractmethod
    def split_data(self, items: List[Any], train_size: int, test_size: int) -> Tuple[List[Any], List[Any]]:
        pass


class DataExporterInterface(ABC):
    """Interface for data export operations"""
    
    @abstractmethod
    def export_data(self, data: Any, filename: str) -> None:
        pass


# Core Data Models (Single Responsibility Principle)
class Item:
    """Represents a product item with price estimation capabilities"""
    
    def __init__(self, datapoint: Dict[str, Any], price: float):
        from transformers import AutoTokenizer
        
        if not hasattr(Item, 'tokenizer'):
            Item.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        
        self.title = datapoint.get("title", "")
        self.description = str(datapoint.get("description", ""))
        self.features = str(datapoint.get("features", ""))
        self.details = str(datapoint.get("details", ""))
        self.category = datapoint.get("main_category", "Unknown")
        self.price = price
        
        # Create combined text and truncate to 160 tokens
        combined_text = f"{self.title} {self.description} {self.features} {self.details}"
        tokens = Item.tokenizer.encode(combined_text, max_length=160, truncation=True)
        self.text = Item.tokenizer.decode(tokens, skip_special_tokens=True)
        self.token_count = len(tokens)
        
        # Determine if item should be included (minimum 50 characters)
        self.include = len(self.text) >= 50 and 1 <= price <= 999
        
        # Create training prompt
        self.prompt = f"Estimate the price of this item: {self.text}\n\nPrice is ${price:.0f}"
    
    def test_prompt(self) -> str:
        """Generate test prompt without revealing the price"""
        return f"Estimate the price of this item: {self.text} to the nearest dollar\n\nPrice is $"


class DatasetConfig:
    """Configuration for dataset processing (Single Responsibility)"""
    
    def __init__(self):
        self.categories = [
            "Automotive", "Electronics", "Office_Products", 
            "Tools_and_Home_Improvement", "Cell_Phones_and_Accessories",
            "Toys_and_Games", "Appliances", "Musical_Instruments"
        ]
        self.price_range = (1, 999)
        self.train_size = 400_000
        self.test_size = 2_000
        self.random_seed = 42


# Concrete Implementations
class HuggingFaceDataLoader(DataLoaderInterface):
    """Loads data from HuggingFace datasets (Single Responsibility)"""
    
    def __init__(self):
        self._setup_environment()
    
    def _setup_environment(self) -> None:
        """Setup environment variables and authentication"""
        load_dotenv(override=True)
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            login(hf_token, add_to_git_credential=True)
    
    def load_dataset(self, category: str) -> Any:
        """Load dataset for specific category"""
        try:
            return load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023", 
                f"raw_meta_{category}", 
                split="full", 
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Error loading {category}: {e}")
            return None


class ItemLoader:
    """Utility class for loading items from datasets"""
    
    def __init__(self, category: str, data_loader: DataLoaderInterface):
        self.category = category
        self.data_loader = data_loader
    
    def load(self) -> List[Item]:
        """Load and convert dataset to Item objects"""
        dataset = self.data_loader.load_dataset(self.category)
        if not dataset:
            return []
        
        items = []
        for datapoint in dataset:
            try:
                price = float(datapoint.get("price", 0))
                if price > 0:
                    item = Item(datapoint, price)
                    if item.include:
                        items.append(item)
            except (ValueError, TypeError):
                continue
        
        print(f"Loaded {len(items):,} items from {self.category}")
        return items


class BalancedDataCurator(DataCuratorInterface):
    """Curates data to create balanced price distribution (Single Responsibility)"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
    
    def curate_items(self, items: List[Item]) -> List[Item]:
        """Create balanced dataset with improved price distribution"""
        # Group items by rounded price
        slots = defaultdict(list)
        for item in items:
            slots[round(item.price)].append(item)
        
        # Set random seeds for reproducibility
        np.random.seed(self.config.random_seed)
        random.seed(self.config.random_seed)
        
        sample = []
        for price in range(1, 1000):
            slot = slots[price]
            if price >= 240:
                # Include all expensive items
                sample.extend(slot)
            elif len(slot) <= 1200:
                # Include all items if slot is small
                sample.extend(slot)
            else:
                # Weighted sampling to reduce automotive bias
                weights = np.array([1 if item.category == 'Automotive' else 5 for item in slot])
                weights = weights / np.sum(weights)
                selected_indices = np.random.choice(len(slot), size=1200, replace=False, p=weights)
                selected = [slot[i] for i in selected_indices]
                sample.extend(selected)
        
        print(f"Curated dataset: {len(sample):,} items")
        return sample


class RandomDataSplitter(DataSplitterInterface):
    """Splits data randomly into train/test sets (Single Responsibility)"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
    
    def split_data(self, items: List[Item], train_size: int, test_size: int) -> Tuple[List[Item], List[Item]]:
        """Split items into training and test sets"""
        random.seed(self.random_seed)
        random.shuffle(items)
        
        train = items[:train_size]
        test = items[train_size:train_size + test_size]
        
        print(f"Split data: {len(train):,} training, {len(test):,} test items")
        return train, test


class PickleDataExporter(DataExporterInterface):
    """Exports data to pickle files (Single Responsibility)"""
    
    def export_data(self, data: Any, filename: str) -> None:
        """Export data to pickle file"""
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        print(f"Exported data to {filename}")


class HuggingFaceDataExporter(DataExporterInterface):
    """Exports data to HuggingFace format (Single Responsibility)"""
    
    def export_data(self, data: Tuple[List[Item], List[Item]], dataset_name: str) -> None:
        """Export train/test data to HuggingFace dataset"""
        train_data, test_data = data
        
        train_prompts = [item.prompt for item in train_data]
        train_prices = [item.price for item in train_data]
        test_prompts = [item.test_prompt() for item in test_data]
        test_prices = [item.price for item in test_data]
        
        train_dataset = Dataset.from_dict({"text": train_prompts, "price": train_prices})
        test_dataset = Dataset.from_dict({"text": test_prompts, "price": test_prices})
        
        dataset = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })
        
        # Uncomment to push to hub
        # dataset.push_to_hub(dataset_name, private=True)
        print(f"Prepared dataset for HuggingFace: {dataset_name}")


class DataVisualizer:
    """Handles data visualization (Single Responsibility)"""
    
    @staticmethod
    def plot_price_distribution(items: List[Item], title: str = "Price Distribution") -> None:
        """Plot price distribution histogram"""
        prices = [item.price for item in items]
        
        plt.figure(figsize=(15, 6))
        plt.title(f"{title}: Avg ${np.mean(prices):.2f}, Max ${max(prices):,.2f}")
        plt.xlabel('Price ($)')
        plt.ylabel('Count')
        plt.hist(prices, bins=range(0, 1000, 10), color="darkblue", alpha=0.7)
        plt.show()
    
    @staticmethod
    def plot_category_distribution(items: List[Item], title: str = "Category Distribution") -> None:
        """Plot category distribution bar chart"""
        category_counts = Counter(item.category for item in items)
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        
        plt.figure(figsize=(15, 6))
        plt.bar(categories, counts, color="goldenrod")
        plt.title(title)
        plt.xlabel('Categories')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels
        for i, v in enumerate(counts):
            plt.text(i, v, f"{v:,}", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()


# Main Data Processing Pipeline (Open/Closed Principle - extensible)
class DataProcessor:
    """Main data processing pipeline orchestrator"""
    
    def __init__(self, 
                 data_loader: DataLoaderInterface,
                 data_curator: DataCuratorInterface,
                 data_splitter: DataSplitterInterface,
                 data_exporter: DataExporterInterface,
                 config: DatasetConfig):
        self.data_loader = data_loader
        self.data_curator = data_curator
        self.data_splitter = data_splitter
        self.data_exporter = data_exporter
        self.config = config
        self.visualizer = DataVisualizer()
    
    def load_all_categories(self) -> List[Item]:
        """Load items from all configured categories"""
        all_items = []
        
        for category in self.config.categories:
            loader = ItemLoader(category, self.data_loader)
            items = loader.load()
            all_items.extend(items)
        
        print(f"Total items loaded: {len(all_items):,}")
        return all_items
    
    def process_full_pipeline(self) -> Tuple[List[Item], List[Item]]:
        """Execute the complete data processing pipeline"""
        print("Starting data processing pipeline...")
        
        # Load data
        items = self.load_all_categories()
        
        # Visualize raw data
        self.visualizer.plot_price_distribution(items, "Raw Data Price Distribution")
        self.visualizer.plot_category_distribution(items, "Raw Data Category Distribution")
        
        # Curate data
        curated_items = self.data_curator.curate_items(items)
        
        # Visualize curated data
        self.visualizer.plot_price_distribution(curated_items, "Curated Data Price Distribution")
        self.visualizer.plot_category_distribution(curated_items, "Curated Data Category Distribution")
        
        # Split data
        train_data, test_data = self.data_splitter.split_data(
            curated_items, 
            self.config.train_size, 
            self.config.test_size
        )
        
        # Export data
        self.data_exporter.export_data(train_data, 'train.pkl')
        self.data_exporter.export_data(test_data, 'test.pkl')
        
        print("Data processing pipeline completed!")
        return train_data, test_data


# Factory Pattern for easy instantiation
class DataProcessorFactory:
    """Factory for creating data processor with default configuration"""
    
    @staticmethod
    def create_default_processor() -> DataProcessor:
        """Create data processor with default implementations"""
        config = DatasetConfig()
        data_loader = HuggingFaceDataLoader()
        data_curator = BalancedDataCurator(config)
        data_splitter = RandomDataSplitter(config.random_seed)
        data_exporter = PickleDataExporter()
        
        return DataProcessor(
            data_loader=data_loader,
            data_curator=data_curator,
            data_splitter=data_splitter,
            data_exporter=data_exporter,
            config=config
        )


# Usage Example
if __name__ == "__main__":
    # Create and run data processor
    processor = DataProcessorFactory.create_default_processor()
    train_data, test_data = processor.process_full_pipeline()
    
    print(f"Processing complete: {len(train_data):,} train, {len(test_data):,} test items")
