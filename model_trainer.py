import os
import re
import math
import json
import random
import pickle
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import Counter
import matplotlib.pyplot as plt

# ML Libraries
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# API Libraries
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv


# Abstract Interfaces (Dependency Inversion Principle)
class ModelInterface(ABC):
    """Interface for all price prediction models"""
    
    @abstractmethod
    def train(self, train_data: List[Any]) -> None:
        pass
    
    @abstractmethod
    def predict(self, item: Any) -> float:
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        pass


class EvaluatorInterface(ABC):
    """Interface for model evaluation"""
    
    @abstractmethod
    def evaluate(self, model: ModelInterface, test_data: List[Any]) -> Dict[str, float]:
        pass


class FeatureExtractorInterface(ABC):
    """Interface for feature extraction"""
    
    @abstractmethod
    def extract_features(self, item: Any) -> Dict[str, Any]:
        pass


# Core Classes (Single Responsibility Principle)
class ModelMetrics:
    """Handles calculation of evaluation metrics"""
    
    @staticmethod
    def calculate_metrics(predictions: List[float], actual: List[float]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        errors = [abs(pred - act) for pred, act in zip(predictions, actual)]
        sles = [(math.log(act + 1) - math.log(pred + 1)) ** 2 for pred, act in zip(predictions, actual)]
        
        avg_error = sum(errors) / len(errors)
        rmsle = math.sqrt(sum(sles) / len(sles))
        
        # Calculate hit rate (within 20% or $40)
        hits = sum(1 for error, actual_price in zip(errors, actual) 
                  if error < 40 or error / actual_price < 0.2)
        hit_rate = hits / len(errors) * 100
        
        return {
            'average_error': avg_error,
            'rmsle': rmsle,
            'hit_rate': hit_rate,
            'total_predictions': len(predictions)
        }


class TestResult:
    """Encapsulates test results for a model"""
    
    def __init__(self, model_name: str, predictions: List[float], 
                 actual: List[float], items: List[Any]):
        self.model_name = model_name
        self.predictions = predictions
        self.actual = actual
        self.items = items
        self.metrics = ModelMetrics.calculate_metrics(predictions, actual)
    
    def get_colors(self) -> List[str]:
        """Get color coding for predictions based on accuracy"""
        colors = []
        for pred, act in zip(self.predictions, self.actual):
            error = abs(pred - act)
            if error < 40 or error / act < 0.2:
                colors.append("green")
            elif error < 80 or error / act < 0.4:
                colors.append("orange")
            else:
                colors.append("red")
        return colors


# Feature Extractors (Single Responsibility)
class BasicFeatureExtractor(FeatureExtractorInterface):
    """Extracts basic features from items"""
    
    def __init__(self, train_data: List[Any]):
        self._calculate_defaults(train_data)
    
    def _calculate_defaults(self, train_data: List[Any]) -> None:
        """Calculate default values from training data"""
        weights = [self._get_weight(item) for item in train_data]
        weights = [w for w in weights if w is not None]
        self.default_weight = sum(weights) / len(weights) if weights else 5.0
        
        ranks = [self._get_rank(item) for item in train_data]
        ranks = [r for r in ranks if r is not None]
        self.default_rank = sum(ranks) / len(ranks) if ranks else 100000.0
    
    def _get_weight(self, item: Any) -> Optional[float]:
        """Extract weight from item features"""
        try:
            features = json.loads(item.details)
            weight_str = features.get('Item Weight')
            if not weight_str:
                return None
            
            parts = weight_str.split(' ')
            amount = float(parts[0])
            unit = parts[1].lower()
            
            # Convert to pounds
            conversions = {
                "pounds": 1.0,
                "ounces": 1/16,
                "grams": 1/453.592,
                "kilograms": 1/0.453592
            }
            return amount * conversions.get(unit, 1.0)
        except:
            return None
    
    def _get_rank(self, item: Any) -> Optional[float]:
        """Extract sales rank from item features"""
        try:
            features = json.loads(item.details)
            rank_dict = features.get("Best Sellers Rank")
            if rank_dict and isinstance(rank_dict, dict):
                ranks = list(rank_dict.values())
                return sum(ranks) / len(ranks)
        except:
            pass
        return None
    
    def extract_features(self, item: Any) -> Dict[str, Any]:
        """Extract all basic features"""
        weight = self._get_weight(item) or self.default_weight
        rank = self._get_rank(item) or self.default_rank
        text_length = len(item.test_prompt())
        
        # Check if it's a top electronics brand
        try:
            features = json.loads(item.details)
            brand = features.get("Brand", "").lower()
            top_brands = ["hp", "dell", "lenovo", "samsung", "asus", "sony", "canon", "apple", "intel"]
            is_top_brand = 1 if brand in top_brands else 0
        except:
            is_top_brand = 0
        
        return {
            "weight": weight,
            "rank": rank,
            "text_length": text_length,
            "is_top_electronics_brand": is_top_brand
        }


# Baseline Models (Single Responsibility)
class RandomPricerModel(ModelInterface):
    """Random price prediction model"""
    
    def train(self, train_data: List[Any]) -> None:
        random.seed(42)
    
    def predict(self, item: Any) -> float:
        return random.randrange(1, 1000)
    
    def get_name(self) -> str:
        return "Random Pricer"


class ConstantPricerModel(ModelInterface):
    """Constant average price prediction model"""
    
    def __init__(self):
        self.average_price = 0.0
    
    def train(self, train_data: List[Any]) -> None:
        prices = [item.price for item in train_data]
        self.average_price = sum(prices) / len(prices)
    
    def predict(self, item: Any) -> float:
        return self.average_price
    
    def get_name(self) -> str:
        return "Constant Pricer"


class LinearRegressionModel(ModelInterface):
    """Linear regression on basic features"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.feature_extractor = None
    
    def train(self, train_data: List[Any]) -> None:
        self.feature_extractor = BasicFeatureExtractor(train_data)
        
        # Extract features
        features_list = [self.feature_extractor.extract_features(item) for item in train_data]
        df = pd.DataFrame(features_list)
        prices = [item.price for item in train_data]
        
        self.model.fit(df, prices)
    
    def predict(self, item: Any) -> float:
        features = self.feature_extractor.extract_features(item)
        df = pd.DataFrame([features])
        return max(0, self.model.predict(df)[0])
    
    def get_name(self) -> str:
        return "Linear Regression"


class BagOfWordsModel(ModelInterface):
    """Bag of Words + Linear Regression model"""
    
    def __init__(self):
        self.vectorizer = CountVectorizer(max_features=1000, stop_words='english')
        self.model = LinearRegression()
    
    def train(self, train_data: List[Any]) -> None:
        documents = [item.test_prompt() for item in train_data]
        prices = [item.price for item in train_data]
        
        X = self.vectorizer.fit_transform(documents)
        self.model.fit(X, prices)
    
    def predict(self, item: Any) -> float:
        X = self.vectorizer.transform([item.test_prompt()])
        return max(0, self.model.predict(X)[0])
    
    def get_name(self) -> str:
        return "Bag of Words"


class Word2VecModel(ModelInterface):
    """Word2Vec + Linear Regression model"""
    
    def __init__(self):
        self.w2v_model = None
        self.regressor = LinearRegression()
    
    def train(self, train_data: List[Any]) -> None:
        documents = [item.test_prompt() for item in train_data]
        prices = [item.price for item in train_data]
        
        # Train Word2Vec
        processed_docs = [simple_preprocess(doc) for doc in documents]
        self.w2v_model = Word2Vec(sentences=processed_docs, vector_size=400, 
                                 window=5, min_count=1, workers=8)
        
        # Create document vectors
        X = np.array([self._document_vector(doc) for doc in documents])
        self.regressor.fit(X, prices)
    
    def _document_vector(self, doc: str) -> np.ndarray:
        """Convert document to vector by averaging word vectors"""
        doc_words = simple_preprocess(doc)
        word_vectors = [self.w2v_model.wv[word] for word in doc_words 
                       if word in self.w2v_model.wv]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(self.w2v_model.vector_size)
    
    def predict(self, item: Any) -> float:
        doc_vector = self._document_vector(item.test_prompt())
        return max(0, self.regressor.predict([doc_vector])[0])
    
    def get_name(self) -> str:
        return "Word2Vec"


class RandomForestModel(ModelInterface):
    """Random Forest with Word2Vec features"""
    
    def __init__(self):
        self.w2v_model = None
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=8)
    
    def train(self, train_data: List[Any]) -> None:
        documents = [item.test_prompt() for item in train_data]
        prices = [item.price for item in train_data]
        
        # Train Word2Vec
        processed_docs = [simple_preprocess(doc) for doc in documents]
        self.w2v_model = Word2Vec(sentences=processed_docs, vector_size=400, 
                                 window=5, min_count=1, workers=8)
        
        # Create document vectors
        X = np.array([self._document_vector(doc) for doc in documents])
        self.rf_model.fit(X, prices)
    
    def _document_vector(self, doc: str) -> np.ndarray:
        """Convert document to vector by averaging word vectors"""
        doc_words = simple_preprocess(doc)
        word_vectors = [self.w2v_model.wv[word] for word in doc_words 
                       if word in self.w2v_model.wv]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(self.w2v_model.vector_size)
    
    def predict(self, item: Any) -> float:
        doc_vector = self._document_vector(item.test_prompt())
        return max(0, self.rf_model.predict([doc_vector])[0])
    
    def get_name(self) -> str:
        return "Random Forest"


# Frontier Models (Open/Closed Principle - extensible)
class OpenAIModel(ModelInterface):
    """OpenAI GPT model for price prediction"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        load_dotenv(override=True)
        self.client = OpenAI()
        self.model_name = model_name
    
    def train(self, train_data: List[Any]) -> None:
        # Frontier models don't need training
        pass
    
    def _create_messages(self, item: Any) -> List[Dict[str, str]]:
        """Create message format for OpenAI API"""
        system_message = "You estimate prices of items. Reply only with the price, no explanation"
        user_prompt = item.test_prompt().replace(" to the nearest dollar", "").replace("\n\nPrice is $", "")
        
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Price is $"}
        ]
    
    def _extract_price(self, response_text: str) -> float:
        """Extract price from model response"""
        text = response_text.replace('$', '').replace(',', '')
        match = re.search(r"[-+]?\d*\.?\d+", text)
        return float(match.group()) if match else 0
    
    def predict(self, item: Any) -> float:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self._create_messages(item),
                seed=42,
                max_tokens=5
            )
            return self._extract_price(response.choices[0].message.content)
        except Exception as e:
            print(f"Error in OpenAI prediction: {e}")
            return 0
    
    def get_name(self) -> str:
        return f"OpenAI {self.model_name}"


class ClaudeModel(ModelInterface):
    """Anthropic Claude model for price prediction"""
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20240620"):
        load_dotenv(override=True)
        self.client = Anthropic()
        self.model_name = model_name
    
    def train(self, train_data: List[Any]) -> None:
        # Frontier models don't need training
        pass
    
    def _extract_price(self, response_text: str) -> float:
        """Extract price from model response"""
        text = response_text.replace('$', '').replace(',', '')
        match = re.search(r"[-+]?\d*\.?\d+", text)
        return float(match.group()) if match else 0
    
    def predict(self, item: Any) -> float:
        try:
            system_message = "You estimate prices of items. Reply only with the price, no explanation"
            user_prompt = item.test_prompt().replace(" to the nearest dollar", "").replace("\n\nPrice is $", "")
            
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=5,
                system=system_message,
                messages=[
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": "Price is $"}
                ]
            )
            return self._extract_price(response.content[0].text)
        except Exception as e:
            print(f"Error in Claude prediction: {e}")
            return 0
    
    def get_name(self) -> str:
        return f"Claude {self.model_name}"


# Evaluation System (Single Responsibility)
class ModelEvaluator(EvaluatorInterface):
    """Evaluates model performance with comprehensive metrics"""
    
    def __init__(self, test_size: int = 250):
        self.test_size = test_size
    
    def evaluate(self, model: ModelInterface, test_data: List[Any]) -> TestResult:
        """Evaluate model on test data"""
        test_subset = test_data[:self.test_size]
        predictions = []
        actual = []
        
        print(f"Evaluating {model.get_name()}...")
        
        for i, item in enumerate(test_subset):
            pred = model.predict(item)
            predictions.append(pred)
            actual.append(item.price)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(test_subset)} items")
        
        return TestResult(model.get_name(), predictions, actual, test_subset)


class ResultVisualizer:
    """Visualizes evaluation results (Single Responsibility)"""
    
    @staticmethod
    def plot_results(result: TestResult) -> None:
        """Create scatter plot of predictions vs actual prices"""
        colors = result.get_colors()
        color_map = {"green": "green", "orange": "orange", "red": "red"}
        plot_colors = [color_map[c] for c in colors]
        
        plt.figure(figsize=(12, 8))
        max_val = max(max(result.actual), max(result.predictions))
        
        # Perfect prediction line
        plt.plot([0, max_val], [0, max_val], color='deepskyblue', lw=2, alpha=0.6, label='Perfect Prediction')
        
        # Scatter plot
        plt.scatter(result.actual, result.predictions, s=3, c=plot_colors, alpha=0.6)
        
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        
        # Title with metrics
        title = (f"{result.model_name}\n"
                f"Avg Error: ${result.metrics['average_error']:.2f} | "
                f"RMSLE: {result.metrics['rmsle']:.2f} | "
                f"Hit Rate: {result.metrics['hit_rate']:.1f}%")
        plt.title(title)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def compare_models(results: List[TestResult]) -> None:
        """Create comparison chart of multiple models"""
        model_names = [r.model_name for r in results]
        avg_errors = [r.metrics['average_error'] for r in results]
        hit_rates = [r.metrics['hit_rate'] for r in results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Average Error comparison
        ax1.bar(model_names, avg_errors, color='skyblue')
        ax1.set_title('Average Error by Model')
        ax1.set_ylabel('Average Error ($)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Hit Rate comparison
        ax2.bar(model_names, hit_rates, color='lightgreen')
        ax2.set_title('Hit Rate by Model')
        ax2.set_ylabel('Hit Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()


# Main Training Pipeline (Facade Pattern)
class ModelTrainer:
    """Main training and evaluation pipeline"""
    
    def __init__(self):
        self.evaluator = ModelEvaluator()
        self.visualizer = ResultVisualizer()
        self.train_data = None
        self.test_data = None
    
    def load_data(self, train_file: str = 'train.pkl', test_file: str = 'test.pkl') -> None:
        """Load training and test data"""
        with open(train_file, 'rb') as f:
            self.train_data = pickle.load(f)
        
        with open(test_file, 'rb') as f:
            self.test_data = pickle.load(f)
        
        print(f"Loaded {len(self.train_data):,} training and {len(self.test_data):,} test items")
    
    def train_baseline_models(self) -> List[ModelInterface]:
        """Train all baseline models"""
        models = [
            RandomPricerModel(),
            ConstantPricerModel(),
            LinearRegressionModel(),
            BagOfWordsModel(),
            Word2VecModel(),
            RandomForestModel()
        ]
        
        print("Training baseline models...")
        for model in models:
            print(f"Training {model.get_name()}...")
            model.train(self.train_data)
        
        return models
    
    def create_frontier_models(self) -> List[ModelInterface]:
        """Create frontier models (no training needed)"""
        return [
            OpenAIModel("gpt-4o-mini"),
            # OpenAIModel("gpt-4o-2024-08-06"),  # Uncomment if you want to test
            # ClaudeModel()  # Uncomment if you want to test
        ]
    
    def evaluate_models(self, models: List[ModelInterface]) -> List[TestResult]:
        """Evaluate all models and return results"""
        results = []
        
        for model in models:
            try:
                result = self.evaluator.evaluate(model, self.test_data)
                results.append(result)
                self.visualizer.plot_results(result)
            except Exception as e:
                print(f"Error evaluating {model.get_name()}: {e}")
        
        return results
    
    def run_full_evaluation(self) -> List[TestResult]:
        """Run complete model training and evaluation pipeline"""
        if not self.train_data or not self.test_data:
            self.load_data()
        
        # Train baseline models
        baseline_models = self.train_baseline_models()
        
        # Create frontier models
        frontier_models = self.create_frontier_models()
        
        # Combine all models
        all_models = baseline_models + frontier_models
        
        # Evaluate all models
        results = self.evaluate_models(all_models)
        
        # Show comparison
        if results:
            self.visualizer.compare_models(results)
        
        return results


# Usage Example
if __name__ == "__main__":
    # Create and run model trainer
    trainer = ModelTrainer()
    results = trainer.run_full_evaluation()
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    for result in results:
        print(f"{result.model_name:20} | "
              f"Avg Error: ${result.metrics['average_error']:6.2f} | "
              f"RMSLE: {result.metrics['rmsle']:5.2f} | "
              f"Hit Rate: {result.metrics['hit_rate']:5.1f}%")
