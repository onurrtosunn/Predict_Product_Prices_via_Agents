"""Visualization utilities for data analysis."""

from typing import List
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from items import Item


class DataVisualizer:
    """Handles visualization of product data."""
    
    def __init__(self, figsize: tuple = (15, 6)):
        self.figsize = figsize
    
    def plot_price_distribution(
        self,
        items: List[Item],
        title: str = "Price Distribution",
        bins: range = range(0, 1000, 10)
    ) -> None:
        """Plot histogram of price distribution."""
        prices = [item.price for item in items]
        
        plt.figure(figsize=self.figsize)
        plt.title(f"{title}\nAvg: ${np.mean(prices):.2f} | Max: ${np.max(prices):.2f}")
        plt.xlabel('Price ($)')
        plt.ylabel('Count')
        plt.hist(prices, rwidth=0.7, color='darkblue', bins=bins)
        plt.tight_layout()
        plt.show()
    
    def plot_token_distribution(
        self,
        items: List[Item],
        title: str = "Token Distribution",
        bins: range = range(0, 300, 10)
    ) -> None:
        """Plot histogram of token count distribution."""
        tokens = [item.token_count for item in items]
        
        plt.figure(figsize=self.figsize)
        plt.title(f"{title}\nAvg: {np.mean(tokens):.1f} | Max: {np.max(tokens)}")
        plt.xlabel('Tokens')
        plt.ylabel('Count')
        plt.hist(tokens, rwidth=0.7, color='green', bins=bins)
        plt.tight_layout()
        plt.show()
    
    def plot_category_bar(self, items: List[Item]) -> None:
        """Plot bar chart of category distribution."""
        category_counts = Counter(item.category for item in items)
        categories = list(category_counts.keys())
        counts = [category_counts[cat] for cat in categories]
        
        plt.figure(figsize=self.figsize)
        plt.bar(categories, counts, color='goldenrod')
        plt.title('Category Distribution')
        plt.xlabel('Categories')
        plt.ylabel('Count')
        plt.xticks(rotation=30, ha='right')
        
        for i, count in enumerate(counts):
            plt.text(i, count, f"{count:,}", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_category_pie(self, items: List[Item]) -> None:
        """Plot donut chart of category distribution."""
        category_counts = Counter(item.category for item in items)
        categories = list(category_counts.keys())
        counts = [category_counts[cat] for cat in categories]
        
        plt.figure(figsize=(12, 10))
        plt.pie(counts, labels=categories, autopct='%1.0f%%', startangle=90)
        
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        
        plt.title('Category Distribution')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
    
    def plot_comprehensive_analysis(self, items: List[Item], name: str = "Dataset") -> None:
        """Generate all visualizations for dataset."""
        print(f"\n{'='*60}")
        print(f"{name} Visualization")
        print(f"{'='*60}\n")
        
        self.plot_price_distribution(items, f"{name} - Price Distribution")
        self.plot_token_distribution(items, f"{name} - Token Distribution")
        self.plot_category_bar(items)
        self.plot_category_pie(items)
