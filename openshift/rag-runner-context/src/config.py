"""
Incoming: config/defaults.yaml --- {YAML configuration}
Processing: config loading + env setup --- {2 jobs: parse YAML, set env vars}
Outgoing: Config singleton + env vars --- {typed config access, cache paths set}

Centralized Configuration
-------------------------
Single source of truth for all configurable values.

Features:
- Loads config/defaults.yaml at import time
- Expands ${VAR} and ${VAR:default} syntax with environment variables
- Provides typed dot-notation access: config.models.bge.name
- Sets up cache environment variables for ML libraries
- PROJECT_ROOT auto-detected from this file's location

STRICT: No fallbacks. Missing config raises ConfigError immediately.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


class ConfigDict:
    """
    Dict wrapper providing dot-notation access.
    
    Allows: config.models.bge.name
    Instead of: config["models"]["bge"]["name"]
    """
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data
        
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigDict(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key: str) -> Any:
        """Get item, wrapping dicts in ConfigDict for consistent access."""
        value = self._data[key]
        if isinstance(value, dict):
            return ConfigDict(value)
        return value
    
    def __contains__(self, key: str) -> bool:
        return key in self._data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item with default, wrapping dicts in ConfigDict."""
        value = self._data.get(key, default)
        if isinstance(value, dict):
            return ConfigDict(value)
        return value
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        """Return values, wrapping dicts in ConfigDict."""
        for value in self._data.values():
            if isinstance(value, dict):
                yield ConfigDict(value)
            else:
                yield value
    
    def items(self):
        """Return items, wrapping dict values in ConfigDict."""
        for key, value in self._data.items():
            if isinstance(value, dict):
                yield key, ConfigDict(value)
            else:
                yield key, value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to plain dict (recursive)."""
        result = {}
        for key, value in self._data.items():
            if isinstance(value, ConfigDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __repr__(self) -> str:
        return f"ConfigDict({self._data})"


class Config:
    """
    Configuration manager with environment variable expansion.
    
    Usage:
        from src.config import config
        
        print(config.models.bge.name)  # "BAAI/bge-base-en-v1.5"
        print(config.paths.cache_root)  # Expanded path
    """
    
    _instance: Optional['Config'] = None
    _initialized: bool = False
    
    # Environment variable expansion pattern: ${VAR} or ${VAR:default}
    ENV_PATTERN = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')
    
    def __new__(cls) -> 'Config':
        """Singleton pattern - one config instance per process."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize config (only runs once due to singleton)."""
        if Config._initialized:
            return
        
        # Determine project root (parent of src/)
        self._src_dir = Path(__file__).parent
        self._project_root = self._src_dir.parent
        
        # Set PROJECT_ROOT in environment for expansion
        os.environ.setdefault('PROJECT_ROOT', str(self._project_root))
        
        # Load configuration
        self._config_path = self._project_root / 'config' / 'defaults.yaml'
        self._raw_config = self._load_yaml()
        self._config = self._expand_env_vars(self._raw_config)
        
        # Create typed accessor
        self._data = ConfigDict(self._config)
        
        # Setup environment (cache paths, threading)
        self._setup_environment()
        
        Config._initialized = True
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not self._config_path.exists():
            raise ConfigError(
                f"Configuration file not found: {self._config_path}\n"
                f"Expected at: config/defaults.yaml"
            )
        
        with open(self._config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _expand_env_vars(self, obj: Any) -> Any:
        """
        Recursively expand environment variables in config values.
        
        Supports:
        - ${VAR} - expands to env var, raises if not set
        - ${VAR:default} - expands to env var or default if not set
        """
        if isinstance(obj, dict):
            return {k: self._expand_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return self._expand_string(obj)
        else:
            return obj
    
    def _expand_string(self, s: str) -> str:
        """Expand environment variables in a string."""
        def replace_match(match):
            var_name = match.group(1)
            default_value = match.group(2)
            
            value = os.environ.get(var_name)
            if value is not None:
                return value
            elif default_value is not None:
                return default_value
            else:
                raise ConfigError(
                    f"Environment variable '{var_name}' not set and no default provided.\n"
                    f"Set it or provide default in config: ${{{var_name}:default_value}}"
                )
        
        return self.ENV_PATTERN.sub(replace_match, s)
    
    def _setup_environment(self) -> None:
        """
        Set up environment variables for ML libraries.
        
        This replaces the old cache_config.py functionality.
        Must be called BEFORE importing transformers, sentence_transformers, pyserini.
        """
        cache_root = Path(self._config['paths']['cache_root'])
        cache_root.mkdir(exist_ok=True)
        
        # Fix FAISS/Java threading crash on Apple Silicon
        os.environ.setdefault('OMP_NUM_THREADS', str(self._config['processing']['threads']['omp']))
        os.environ.setdefault('MKL_NUM_THREADS', str(self._config['processing']['threads']['mkl']))
        
        # Pyserini index cache
        os.environ.setdefault('PYSERINI_CACHE', str(cache_root / 'pyserini'))
        
        # HuggingFace model cache
        os.environ.setdefault('HF_HOME', str(cache_root / 'huggingface'))
        os.environ.setdefault('HF_DATASETS_CACHE', str(cache_root / 'huggingface' / 'datasets'))
        
        # Sentence Transformers cache
        os.environ.setdefault('SENTENCE_TRANSFORMERS_HOME', str(cache_root / 'sentence_transformers'))
        
        # PyTorch hub cache
        os.environ.setdefault('TORCH_HOME', str(cache_root / 'torch'))
        
        # Java settings
        java_opts = self._config['processing']['java']['tool_options']
        os.environ.setdefault('JAVA_TOOL_OPTIONS', java_opts)
        
        # Tokenizers parallelism
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')
        
        # Fix for duplicate OpenMP libraries on macOS
        os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    
    # =========================================================================
    # Property accessors (delegate to ConfigDict)
    # =========================================================================
    
    @property
    def paths(self) -> ConfigDict:
        return self._data.paths
    
    @property
    def datasets(self) -> ConfigDict:
        return self._data.datasets
    
    @property
    def indexes(self) -> ConfigDict:
        return self._data.indexes
    
    @property
    def models(self) -> ConfigDict:
        return self._data.models
    
    @property
    def processing(self) -> ConfigDict:
        return self._data.processing
    
    @property
    def qpp(self) -> ConfigDict:
        return self._data.qpp
    
    @property
    def fusion(self) -> ConfigDict:
        return self._data.fusion
    
    @property
    def evaluation(self) -> ConfigDict:
        return self._data.evaluation
    
    @property
    def training(self) -> ConfigDict:
        return self._data.training
    
    @property
    def generation(self) -> ConfigDict:
        return self._data.generation
    
    @property
    def project_root(self) -> Path:
        return self._project_root
    
    @property
    def cache_root(self) -> Path:
        return Path(self._config['paths']['cache_root'])
    
    # =========================================================================
    # Helper methods
    # =========================================================================
    
    def get_dataset_config(self, dataset: str) -> ConfigDict:
        """Get configuration for a specific dataset."""
        if dataset not in self._config['datasets']:
            supported = self._config['datasets'].get('supported', [])
            raise ConfigError(f"Unknown dataset: {dataset}. Supported: {supported}")
        return ConfigDict(self._config['datasets'][dataset])
    
    def get_index_hash(self, index_type: str, dataset: str) -> str:
        """Get the hash/path for a specific index type and dataset."""
        index_config = self._config['indexes']['pyserini'].get(index_type)
        if not index_config:
            raise ConfigError(f"Unknown index type: {index_type}")
        
        hashes = index_config.get('hashes', {})
        if dataset not in hashes:
            raise ConfigError(f"No hash for {index_type}/{dataset}")
        
        return hashes[dataset]
    
    def get_index_name(self, index_type: str, dataset: str) -> str:
        """Get the Pyserini index name for a specific type and dataset."""
        index_config = self._config['indexes']['pyserini'].get(index_type)
        if not index_config:
            raise ConfigError(f"Unknown index type: {index_type}")
        
        template = index_config.get('name_template', '')
        return template.format(dataset=dataset)
    
    def get_qpp_index(self, method_name: str) -> int:
        """Get the index for a QPP method name."""
        method_index = self._config['qpp']['method_index']
        
        # Handle case variations
        for name, idx in method_index.items():
            if name.lower() == method_name.lower():
                return idx
        
        # Special case: "fusion" means average all
        if method_name.lower() == 'fusion':
            return -1
        
        raise ConfigError(
            f"Unknown QPP method: {method_name}. "
            f"Valid: {list(method_index.keys())} or 'fusion'"
        )
    
    def get_beir_path(self, dataset: str) -> Path:
        """Get the BEIR dataset path."""
        base = Path(self._config['paths']['beir_datasets'])
        return base / dataset
    
    def get_qrels_path(self, dataset: str) -> Path:
        """Get the qrels file path for a dataset."""
        beir_path = self.get_beir_path(dataset)
        dataset_config = self.get_dataset_config(dataset)
        return beir_path / dataset_config.qrels_file
    
    def to_dict(self) -> Dict[str, Any]:
        """Get the full configuration as a dict."""
        return self._config.copy()
    
    def __getitem__(self, key: str) -> Any:
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self._config


# =============================================================================
# Module-level singleton
# =============================================================================

# Create singleton instance on import
# This also triggers environment setup
config = Config()

# Expose cache_root for backward compatibility with cache_config.py imports
CACHE_ROOT = config.cache_root
PROJECT_ROOT = config.project_root


def setup_cache() -> Path:
    """
    Backward compatibility function.
    
    The cache is now set up automatically on import.
    This function exists only for code that explicitly calls it.
    """
    return config.cache_root


# =============================================================================
# Utility functions using config
# =============================================================================

def get_device() -> str:
    """
    Get the best available compute device.
    
    Returns: "mps" (Apple Silicon), "cuda", or "cpu"
    
    This is a utility function that was duplicated across multiple files.
    Now centralized here.
    """
    import torch
    
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def ensure_pyterrier_init():
    """
    Lazy PyTerrier initialization with memory limits from config.
    
    This was duplicated in bm25.py, bm25_tct.py, bm25_monot5.py.
    Now centralized here.
    """
    import pyterrier as pt
    
    if not pt.started():
        mem_mb = config.processing.threads.pyterrier_mem_mb
        try:
            pt.init(boot_packages=[], mem=mem_mb)
        except TypeError:
            # Older PyTerrier version doesn't support mem parameter
            pt.init()
    
    return pt


def get_model_safe_name(model: str) -> str:
    """
    Convert model name to filesystem-safe name.
    
    Moved from data_utils.py (doesn't belong there).
    """
    return model.replace("/", "_").replace(":", "_")


def detect_dataset(path: str) -> str:
    """
    Detect dataset name from path.
    
    Moved from data_utils.py (doesn't belong there).
    """
    path_lower = path.lower()
    for dataset in config.datasets.supported:
        if dataset in path_lower:
            return dataset
    return "unknown"
