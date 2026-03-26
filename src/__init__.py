"""
Source module initialization.

Imports all main classes and functions from the visual assistant pipeline.
"""

try:
    from .detector import YOLODetector
except ImportError:
    YOLODetector = None

try:
    from .segmentor import MobileSAMSegmentor
except ImportError:
    MobileSAMSegmentor = None

try:
    from .depth_estimator import DepthEstimator
except ImportError:
    DepthEstimator = None

try:
    from .llm_generator import LLMGenerator
except ImportError:
    LLMGenerator = None

try:
    from .pipeline import VisualAssistantPipeline
except ImportError:
    VisualAssistantPipeline = None

from .visualizer import Visualizer
from .data_loader import DataLoader, CustomDataLoader
from .evaluation import Evaluator

__all__ = [
    'YOLODetector',
    'MobileSAMSegmentor',
    'DepthEstimator',
    'LLMGenerator',
    'VisualAssistantPipeline',
    'Visualizer',
    'DataLoader',
    'CustomDataLoader',
    'Evaluator',
]

