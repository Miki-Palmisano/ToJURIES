"""to_juries: evaluate two reports against an expert reference with LLM judges."""

try:  # keys may live in a .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from .analysis.pipeline import analyze
from .config import ConfigError
from .dataset import Pair, PairDataset
from .evaluator import Evaluator, Run
from .prompt import Prompt
from .judje import AzureOpenAI, Gemini, LLMResponse, Provider, ProviderError

__version__ = "0.3.0"
__all__ = ["Evaluator", "Run", "PairDataset", "Pair", "Prompt", "AzureOpenAI", "Gemini",
           "Provider", "LLMResponse", "ProviderError", "ConfigError", "analyze"]
