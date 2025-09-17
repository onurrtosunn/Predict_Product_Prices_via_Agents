import os
import re
import math
import torch
import wandb
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm

import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    set_seed
)
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """
    Configuration for model parameters
    """
    base_model: str = "meta-llama/Meta-Llama-3.1-8B"
    hf_user: str = "HF_USERNAME"
    dataset_name: str = None
    max_sequence_length: int = 182
    quant_4_bit: bool = True
    
    def __post_init__(self):
        if self.dataset_name is None:
            self.dataset_name = f"{self.hf_user}/pricer-data"


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA parameters
    """
    lora_r: int = 32
    lora_alpha: int = 64
    target_modules: List[str] = None
    lora_dropout: float = 0.1
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


@dataclass
class TrainingConfig:
    """
    Configuration for training parameters
    """
    epochs: int = 1
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_scheduler_type: str = 'cosine'
    warmup_ratio: float = 0.03
    optimizer: str = "paged_adamw_32bit"
    save_steps: int = 2000
    log_to_wandb: bool = True
    project_name: str = "pricer"


class QuantizationAnalyzer:
    """
    Analyzes different quantization methods and their memory footprints
    """
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        
    def compare_quantization_methods(self) -> Dict[str, float]:
        """
        Compare memory footprints of different quantization methods
        """
        results = {}
        
        # No quantization
        print("Loading model without quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model, 
            device_map="auto"
        )
        results['no_quant'] = model.get_memory_footprint() / 1e9
        del model
        torch.cuda.empty_cache()
        
        # 8-bit quantization
        print("Loading model with 8-bit quantization...")
        quant_config_8bit = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model,
            quantization_config=quant_config_8bit,
            device_map="auto"
        )
        results['8bit'] = model.get_memory_footprint() / 1e9
        del model
        torch.cuda.empty_cache()
        
        # 4-bit quantization
        print("Loading model with 4-bit quantization...")
        quant_config_4bit = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model,
            quantization_config=quant_config_4bit,
            device_map="auto"
        )
        results['4bit'] = model.get_memory_footprint() / 1e9
        del model
        torch.cuda.empty_cache()
        
        return results
    
    def calculate_lora_parameters(self, lora_config: LoRAConfig) -> Dict[str, int]:
        """
        Calculate LoRA parameter counts
        """
        # Matrix dimensions for Llama 3.1 8B
        lora_q_proj = 4096 * lora_config.lora_r + 4096 * lora_config.lora_r
        lora_k_proj = 4096 * lora_config.lora_r + 1024 * lora_config.lora_r
        lora_v_proj = 4096 * lora_config.lora_r + 1024 * lora_config.lora_r
        lora_o_proj = 4096 * lora_config.lora_r + 4096 * lora_config.lora_r
        
        lora_layer = lora_q_proj + lora_k_proj + lora_v_proj + lora_o_proj
        total_params = lora_layer * 32  # 32 layers
        size_mb = (total_params * 4) / 1_000_000
        
        return {
            'total_params': total_params,
            'size_mb': size_mb,
            'per_layer': lora_layer
        }


class TokenizerAnalyzer:
    """
    Analyzes different tokenizers for number handling
    """
    
    MODELS = {
        "LLAMA_3_1": "meta-llama/Meta-Llama-3.1-8B",
        "QWEN_2_5": "Qwen/Qwen2.5-7B",
        "GEMMA_2": "google/gemma-2-9b",
        "PHI_3": "microsoft/Phi-3-medium-4k-instruct"
    }
    
    def investigate_tokenizer(self, model_name: str) -> Dict[int, List[int]]:
        """
        Investigate how a tokenizer handles numbers
        """
        print(f"Investigating tokenizer for {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        results = {}
        for number in [0, 1, 10, 100, 999, 1000]:
            tokens = tokenizer.encode(str(number), add_special_tokens=False)
            results[number] = tokens
            print(f"The tokens for {number}: {tokens}")
        
        return results
    
    def compare_all_tokenizers(self) -> Dict[str, Dict[int, List[int]]]:
        """
        Compare all available tokenizers
        """
        results = {}
        for name, model_name in self.MODELS.items():
            try:
                results[name] = self.investigate_tokenizer(model_name)
            except Exception as e:
                print(f"Error with {name}: {e}")
                results[name] = None
        return results


class PriceExtractor:
    """
    Utility class for extracting prices from model outputs
    """
    
    @staticmethod
    def extract_price(text: str) -> float:
        """
        Extract price from model output text
        """
        if "Price is $" in text:
            contents = text.split("Price is $")[1]
            contents = contents.replace(',', '').replace('$', '')
            match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
            return float(match.group()) if match else 0
        return 0


class ModelEvaluator:
    """
    Evaluates model performance with comprehensive metrics
    """
    
    # Color codes for output
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}
    
    def __init__(self, predictor, data, title: Optional[str] = None, size: int = 250):
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = size
        self.guesses = []
        self.truths = []
        self.errors = []
        self.sles = []
        self.colors = []
    
    def color_for(self, error: float, truth: float) -> str:
        """
        Determine color based on error magnitude
        """
        if error < 40 or error / truth < 0.2:
            return "green"
        elif error < 80 or error / truth < 0.4:
            return "orange"
        else:
            return "red"
    
    def run_datapoint(self, i: int):
        """
        Evaluate a single datapoint
        """
        datapoint = self.data[i]
        guess = self.predictor(datapoint["text"])
        truth = datapoint["price"]
        error = abs(guess - truth)
        log_error = math.log(truth + 1) - math.log(guess + 1)
        sle = log_error ** 2
        color = self.color_for(error, truth)
        title = datapoint["text"].split("\n\n")[1][:20] + "..."
        
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.sles.append(sle)
        self.colors.append(color)
        
        print(f"{self.COLOR_MAP[color]}{i+1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} "
              f"Error: ${error:,.2f} SLE: {sle:,.2f} Item: {title}{self.RESET}")
    
    def chart(self, title: str):
        """
        Create scatter plot of predictions vs truth
        """
        plt.figure(figsize=(12, 8))
        max_val = max(max(self.truths), max(self.guesses))
        plt.plot([0, max_val], [0, max_val], color='deepskyblue', lw=2, alpha=0.6)
        plt.scatter(self.truths, self.guesses, s=3, c=self.colors)
        plt.xlabel('Ground Truth')
        plt.ylabel('Model Estimate')
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.title(title)
        plt.show()
    
    def report(self):
        """
        Generate comprehensive evaluation report
        """
        average_error = sum(self.errors) / self.size
        rmsle = math.sqrt(sum(self.sles) / self.size)
        hits = sum(1 for color in self.colors if color == "green")
        title = f"{self.title} Error=${average_error:,.2f} RMSLE={rmsle:,.2f} Hits={hits/self.size*100:.1f}%"
        self.chart(title)
    
    def run(self):
        """
        Run complete evaluation
        """
        for i in range(self.size):
            self.run_datapoint(i)
        self.report()
    
    @classmethod
    def test(cls, function, data, **kwargs):
        """
        Convenience method for quick testing
        """
        cls(function, data, **kwargs).run()


class ProductPricerPipeline:
    """
    Main pipeline class that orchestrates the entire process
    """
    
    def __init__(self, 
                 model_config: ModelConfig,
                 lora_config: LoRAConfig,
                 training_config: TrainingConfig):
        self.model_config = model_config
        self.lora_config = lora_config
        self.training_config = training_config
        self.tokenizer = None
        self.base_model = None
        self.fine_tuned_model = None
        self.dataset = None
        
    def setup_authentication(self, hf_token: str, wandb_key: Optional[str] = None):
        """
        Setup HuggingFace and Weights & Biases authentication
        """
        from huggingface_hub import login
        login(hf_token, add_to_git_credential=True)
        
        if wandb_key and self.training_config.log_to_wandb:
            os.environ["WANDB_API_KEY"] = wandb_key
            wandb.login()
            os.environ["WANDB_PROJECT"] = self.training_config.project_name
            os.environ["WANDB_LOG_MODEL"] = "checkpoint"
            os.environ["WANDB_WATCH"] = "gradients"
    
    def load_dataset(self):
        """
        Load the dataset
        """
        self.dataset = load_dataset(self.model_config.dataset_name)
        return self.dataset
    
    def setup_quantization_config(self) -> BitsAndBytesConfig:
        """
        Setup quantization configuration
        """
        if self.model_config.quant_4_bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
        else:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
    
    def load_base_model(self):
        """
        Load tokenizer and base model
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.base_model, 
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        quant_config = self.setup_quantization_config()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_config.base_model,
            quantization_config=quant_config,
            device_map="auto"
        )
        self.base_model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        
        print(f"Memory footprint: {self.base_model.get_memory_footprint() / 1e6:.1f} MB")
        return self.base_model
    
    def create_prediction_function(self, model=None):
        """
        Create basic prediction function
        """
        if model is None:
            model = self.base_model
            
        def model_predict(prompt: str) -> float:
            set_seed(42)
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
            attention_mask = torch.ones(inputs.shape, device="cuda")
            outputs = model.generate(
                inputs, 
                max_new_tokens=4, 
                attention_mask=attention_mask, 
                num_return_sequences=1
            )
            response = self.tokenizer.decode(outputs[0])
            return PriceExtractor.extract_price(response)
        
        return model_predict
    
    def create_improved_prediction_function(self, model=None, top_k: int = 3):
        """
        Create improved prediction function with weighted averaging
        """
        if model is None:
            model = self.fine_tuned_model or self.base_model
            
        def improved_model_predict(prompt: str, device: str = "cuda") -> float:
            set_seed(42)
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
            attention_mask = torch.ones(inputs.shape, device=device)
            
            with torch.no_grad():
                outputs = model(inputs, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :].to('cpu')
            
            next_token_probs = F.softmax(next_token_logits, dim=-1)
            top_prob, top_token_id = next_token_probs.topk(top_k)
            prices, weights = [], []
            
            for i in range(top_k):
                predicted_token = self.tokenizer.decode(top_token_id[0][i])
                probability = top_prob[0][i]
                try:
                    result = float(predicted_token)
                except ValueError:
                    result = 0.0
                if result > 0:
                    prices.append(result)
                    weights.append(probability)
            
            if not prices:
                return 0.0
            
            total = sum(weights)
            weighted_prices = [price * weight / total for price, weight in zip(prices, weights)]
            return sum(weighted_prices).item()
        
        return improved_model_predict
    
    def evaluate_base_model(self, test_size: int = 250):
        """
        Evaluate base model performance
        """
        if not self.dataset:
            self.load_dataset()
        
        predictor = self.create_prediction_function()
        ModelEvaluator.test(predictor, self.dataset['test'], size=test_size)
    
    def train_model(self, run_name: Optional[str] = None):
        """
        Train the model using QLoRA
        """
        if not self.dataset:
            self.load_dataset()
        
        if not self.base_model:
            self.load_base_model()
        
        # Generate run name
        if run_name is None:
            run_name = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
        
        project_run_name = f"{self.training_config.project_name}-{run_name}"
        hub_model_name = f"{self.model_config.hf_user}/{project_run_name}"
        
        # Initialize wandb if needed
        if self.training_config.log_to_wandb:
            wandb.init(project=self.training_config.project_name, name=run_name)
        
        # Setup data collator
        response_template = "Price is $"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)
        
        # LoRA configuration
        lora_parameters = LoraConfig(
            lora_alpha=self.lora_config.lora_alpha,
            lora_dropout=self.lora_config.lora_dropout,
            r=self.lora_config.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.lora_config.target_modules,
        )
        
        # Training configuration
        train_parameters = SFTConfig(
            output_dir=project_run_name,
            num_train_epochs=self.training_config.epochs,
            per_device_train_batch_size=self.training_config.batch_size,
            per_device_eval_batch_size=1,
            eval_strategy="no",
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            optim=self.training_config.optimizer,
            save_steps=self.training_config.save_steps,
            save_total_limit=10,
            logging_steps=50,
            learning_rate=self.training_config.learning_rate,
            weight_decay=0.001,
            fp16=False,
            bf16=True,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=self.training_config.warmup_ratio,
            group_by_length=True,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            report_to="wandb" if self.training_config.log_to_wandb else None,
            run_name=run_name,
            max_seq_length=self.model_config.max_sequence_length,
            dataset_text_field="text",
            save_strategy="steps",
            hub_strategy="every_save",
            push_to_hub=True,
            hub_model_id=hub_model_name,
            hub_private_repo=True
        )
        
        # Create trainer
        fine_tuning = SFTTrainer(
            model=self.base_model,
            train_dataset=self.dataset['train'],
            peft_config=lora_parameters,
            args=train_parameters,
            data_collator=collator
        )
        
        # Train
        fine_tuning.train()
        
        # Save to hub
        fine_tuning.model.push_to_hub(project_run_name, private=True)
        print(f"Saved to the hub: {project_run_name}")
        
        if self.training_config.log_to_wandb:
            wandb.finish()
        
        return project_run_name
    
    def load_fine_tuned_model(self, model_name: str, revision: Optional[str] = None):
        """
        Load a fine-tuned model
        """
        if not self.base_model:
            self.load_base_model()
        
        if revision:
            self.fine_tuned_model = PeftModel.from_pretrained(
                self.base_model, model_name, revision=revision
            )
        else:
            self.fine_tuned_model = PeftModel.from_pretrained(
                self.base_model, model_name
            )
        
        print(f"Fine-tuned model memory footprint: {self.fine_tuned_model.get_memory_footprint() / 1e6:.1f} MB")
        return self.fine_tuned_model
    
    def evaluate_fine_tuned_model(self, test_size: int = 250, use_improved: bool = True):
        """
        Evaluate fine-tuned model performance
        """
        if not self.dataset:
            self.load_dataset()
        
        if not self.fine_tuned_model:
            raise ValueError("Fine-tuned model not loaded. Use load_fine_tuned_model() first.")
        
        if use_improved:
            predictor = self.create_improved_prediction_function()
        else:
            predictor = self.create_prediction_function(self.fine_tuned_model)
        
        ModelEvaluator.test(predictor, self.dataset['test'], size=test_size)
    
    def run_complete_pipeline(self, hf_token: str, wandb_key: Optional[str] = None, 
                            train: bool = True, evaluate: bool = True):
        """ 
        Run the complete pipeline from start to finish
        """
        print("=== Starting Complete Product Pricer Pipeline ===")
        
        # Setup authentication
        self.setup_authentication(hf_token, wandb_key)
        
        # Load dataset
        print("\n1. Loading dataset...")
        self.load_dataset()
        
        # Load base model
        print("\n2. Loading base model...")
        self.load_base_model()
        
        # Evaluate base model
        print("\n3. Evaluating base model...")
        self.evaluate_base_model()
        
        if train:
            # Train model
            print("\n4. Training model...")
            model_name = self.train_model()
            
            # Load fine-tuned model
            print("\n5. Loading fine-tuned model...")
            self.load_fine_tuned_model(model_name)
        
        if evaluate and self.fine_tuned_model:
            # Evaluate fine-tuned model
            print("\n6. Evaluating fine-tuned model...")
            self.evaluate_fine_tuned_model()
        
        print("\n=== Pipeline Complete ===")


# Example usage and configuration
def create_default_pipeline() -> ProductPricerPipeline:
    """
    Create a pipeline with default configurations
    """
    model_config = ModelConfig()
    lora_config = LoRAConfig()
    training_config = TrainingConfig()
    
    return ProductPricerPipeline(model_config, lora_config, training_config)


def analyze_quantization_methods():
    """
    Analyze different quantization methods
    """
    model_config = ModelConfig()
    analyzer = QuantizationAnalyzer(model_config)
    
    print("Analyzing quantization methods...")
    results = analyzer.compare_quantization_methods()
    
    print("\nQuantization Results:")
    for method, memory_gb in results.items():
        print(f"{method}: {memory_gb:.2f} GB")
    
    return results


def analyze_tokenizers():
    """
    Analyze different tokenizers
    """
    analyzer = TokenizerAnalyzer()
    results = analyzer.compare_all_tokenizers()
    
    print("\nTokenizer Analysis Results:")
    for model_name, token_results in results.items():
        if token_results:
            print(f"\n{model_name}:")
            for number, tokens in token_results.items():
                print(f"  {number}: {tokens}")
    
    return results


if __name__ == "__main__":
    print("Product Pricer Unified Pipeline")
    print("Use create_default_pipeline() to get started")
