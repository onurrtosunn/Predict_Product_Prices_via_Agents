"""Product item representation with tokenization and prompt generation."""

from typing import Optional
from transformers import AutoTokenizer

from config import (
    BASE_MODEL, MIN_TOKENS, MAX_TOKENS, MIN_CHARS, CEILING_CHARS,
    PRICE_PREFIX, PRICE_QUESTION
)
from text_utils import TextCleaner, ContentAggregator

class Item:
    """Curated product datapoint with price and tokenized prompt."""
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    PREFIX = PRICE_PREFIX
    QUESTION = PRICE_QUESTION

    title: str
    price: float
    category: str
    token_count: int = 0
    details: Optional[str]
    prompt: Optional[str] = None
    include: bool = False

    def __init__(self, data, price):
        self.title = data['title']
        self.price = price
        self.parse(data)

    def scrub_details(self) -> str:
        """Clean product details using TextCleaner."""
        return TextCleaner.clean_details(self.details) if self.details else ""

    def scrub(self, text: str) -> str:
        """Clean general text using TextCleaner."""
        return TextCleaner.clean_general_text(text)
    
    def parse(self, data: dict) -> None:
        """Parse and validate datapoint, setting include flag if criteria met."""
        self.details = data.get('details')
        
        contents = ContentAggregator.combine_fields(
            data['title'],
            data.get('description', []),
            data.get('features', []),
            self.details
        )
        
        if len(contents) <= MIN_CHARS:
            return
        
        contents = contents[:CEILING_CHARS]
        text = f"{self.scrub(self.title)}\n{self.scrub(contents)}"
        
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= MIN_TOKENS:
            return
        
        tokens = tokens[:MAX_TOKENS]
        text = self.tokenizer.decode(tokens)
        self.make_prompt(text)
        self.include = True

    def make_prompt(self, text: str) -> None:
        """Generate training prompt with price."""
        rounded_price = round(self.price)
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n{self.PREFIX}{rounded_price}.00"
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))

    def test_prompt(self) -> str:
        """Generate test prompt without price."""
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX

    def __repr__(self) -> str:
        """String representation of Item."""
        return f"<Item: {self.title} = ${self.price:.2f}>"

        

    
    