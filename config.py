"""Configuration constants for product price prediction pipeline."""

# Model Configuration
BASE_MODEL = "Qwen/Qwen3-0.6B"
MIN_TOKENS = 150
MAX_TOKENS = 160
MIN_CHARS = 300
CEILING_CHARS = MAX_TOKENS * 7

# Data Processing
MIN_PRICE = 0.5
MAX_PRICE = 999.49
CHUNK_SIZE = 1000
RANDOM_SEED = 42

# Dataset Split
TRAIN_SIZE = 400_000
TEST_SIZE = 2_000

# Balancing Parameters
PRICE_SLOT_THRESHOLD = 240
MAX_ITEMS_PER_SLOT = 1200
AUTOMOTIVE_WEIGHT = 1.0
OTHER_CATEGORY_WEIGHT = 5.0

# Prompt Templates
PRICE_PREFIX = "Price is $"
PRICE_QUESTION = "How much does this cost to the nearest dollar?"

# Text Cleaning Patterns
REMOVAL_PATTERNS = [
    '"Batteries Included?": "No"',
    '"Batteries Included?": "Yes"',
    '"Batteries Required?": "No"',
    '"Batteries Required?": "Yes"',
    "By Manufacturer",
    "Item",
    "Date First",
    "Package",
    ":",
    "Number of",
    "Best Sellers",
    "Number",
    "Product "
]

# Dataset Categories
DATASET_CATEGORIES = [
    "Automotive",
    "Electronics",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Cell_Phones_and_Accessories",
    "Toys_and_Games",
    "Appliances",
    "Musical_Instruments",
]

DATASET_SOURCE = "McAuley-Lab/Amazon-Reviews-2023"
