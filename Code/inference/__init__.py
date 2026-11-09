"""Prediction from source images without persistent test tensors or features."""
from .direct import InferenceError, PredictionResult, predict
__all__ = ['InferenceError', 'PredictionResult', 'predict']
