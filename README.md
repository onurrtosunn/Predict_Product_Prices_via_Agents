# Product Price Prediction Pipeline

Professional data curation pipeline for training product price prediction models using Amazon Reviews 2023 dataset.

## Architecture

The project follows SOLID principles with clear separation of concerns:

### Core Modules

#### `config.py`
Centralized configuration management for all constants and parameters.
- Model configuration (tokenizer, token limits)
- Data processing parameters (price ranges, chunk sizes)
- Dataset balancing parameters
- Prompt templates

#### `items.py`
Product item representation with tokenization and prompt generation.
- `Item`: Main class for curated product datapoints
- Handles text cleaning, tokenization, and prompt creation
- Validates items against quality criteria

#### `loaders.py`
Dataset loading with parallel processing capabilities.
- `ItemLoader`: Loads and processes HuggingFace datasets
- Parallel chunk processing for performance
- Price validation and filtering

#### `text_utils.py`
Text processing utilities following Single Responsibility Principle.
- `TextCleaner`: Handles text cleaning operations
- `ContentAggregator`: Combines product information from multiple fields

#### `data_utils.py`
Data processing and analysis utilities.
- `DatasetBalancer`: Balances dataset by price slots
- `DatasetSplitter`: Splits data into train/test sets
- `DatasetExporter`: Exports to various formats (HuggingFace, pickle)
- `DatasetAnalyzer`: Calculates and displays statistics

#### `visualization.py`
Visualization utilities for data analysis.
- `DataVisualizer`: Creates plots for price, token, and category distributions
- Supports histograms, bar charts, and pie charts

### Notebooks

#### `product_price_pipeline.ipynb`
Professional notebook with minimal markdown and clean code structure.
- Environment setup
- Single category analysis
- Multi-category loading
- Dataset balancing
- Train/test splitting
- Export and validation

#### `day1.ipynb` (Original)
Original tutorial-style notebook (kept for reference).

## Usage

### Basic Workflow

```python
from items import Item
from loaders import ItemLoader
from data_utils import DatasetBalancer, DatasetSplitter, DatasetExporter
from visualization import DataVisualizer
from config import DATASET_CATEGORIES

# Load datasets
all_items = []
for category in DATASET_CATEGORIES:
    loader = ItemLoader(category)
    items = loader.load(workers=8)
    all_items.extend(items)

# Balance dataset
balancer = DatasetBalancer()
balanced_items = balancer.balance(all_items)

# Split into train/test
splitter = DatasetSplitter()
train_items, test_items = splitter.split(balanced_items)

# Export
DatasetExporter.to_pickle(train_items, 'train.pkl')
DatasetExporter.to_pickle(test_items, 'test.pkl')

# Visualize
visualizer = DataVisualizer()
visualizer.plot_comprehensive_analysis(train_items, "Training Set")
```

### Configuration

Modify `config.py` to adjust:
- Token limits (MIN_TOKENS, MAX_TOKENS)
- Price ranges (MIN_PRICE, MAX_PRICE)
- Dataset split sizes (TRAIN_SIZE, TEST_SIZE)
- Balancing parameters

## Design Principles

### SOLID Compliance

1. **Single Responsibility**: Each class has one clear purpose
   - `TextCleaner` only handles text cleaning
   - `ItemLoader` only handles data loading
   - `DatasetBalancer` only handles balancing

2. **Open/Closed**: Extensible without modification
   - Add new text cleaning strategies by extending `TextCleaner`
   - Add new export formats by extending `DatasetExporter`

3. **Liskov Substitution**: Proper inheritance hierarchy
   - Abstract base classes where appropriate

4. **Interface Segregation**: Focused interfaces
   - Utility classes provide specific functionality
   - No bloated interfaces

5. **Dependency Inversion**: Depend on abstractions
   - Configuration injected via `config.py`
   - Type hints for clear contracts

### Code Quality

- **Type Hints**: All functions have proper type annotations
- **Docstrings**: Clear documentation for all classes and methods
- **Error Handling**: Robust exception handling
- **Logging**: Proper logging instead of print statements
- **DRY**: No code duplication, reusable utilities

## Dataset

**Source**: McAuley-Lab/Amazon-Reviews-2023  
**Categories**: Automotive, Electronics, Office Products, Tools, Cell Phones, Toys, Appliances, Musical Instruments

**Processing**:
- Price range: $0.50 - $999.49
- Token range: 150-160 tokens (content) + prompt overhead
- Minimum content: 300 characters
- Balanced by price slots to reduce skew

## Output

- `train.pkl`: 400,000 training items
- `test.pkl`: 2,000 test items
- HuggingFace DatasetDict format available

## Requirements

```
transformers
datasets
huggingface_hub
tqdm
numpy
matplotlib
python-dotenv
```

## Environment Variables

Create `.env` file:
```
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_key (optional)
ANTHROPIC_API_KEY=your_anthropic_key (optional)
```
