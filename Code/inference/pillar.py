"""Pillar raw-image convenience entry point."""
from .preprocessing import preprocess_direct

def preprocess_nifti(path, runtime):
    return preprocess_direct('Pillar0', path, {'assets': runtime})
