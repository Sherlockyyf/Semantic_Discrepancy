"""Dissecting and Mitigating Semantic Discrepancy in Stable Diffusion for Image-to-Image Translation."""
from .refinement import RefinementConfig, SemanticDiscrepancyRefiner

__version__ = '1.0.0'


def __getattr__(name):  # lazy import, so that the refiner can be used without loading diffusers pipelines
    if name in ('SemanticDiscrepancyPipeline', 'DEFAULT_BASE_MODEL', 'DEFAULT_CONTROLNET'):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(name)


__all__ = ['RefinementConfig', 'SemanticDiscrepancyRefiner', 'SemanticDiscrepancyPipeline']
