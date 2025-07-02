import warnings

# Import all functions from the new visualization module
from vut.visualization import (
    make_video,
    plot_action_segmentation,
    plot_feature,
    plot_features,
    plot_image,
    plot_images,
    plot_metrics,
    plot_palette,
    plot_roc_curve,
    plot_scatter,
)

# Issue deprecation warning when this module is imported
warnings.warn(
    "The 'vut.visualize' module is deprecated and will be removed in v0.2.0. "
    "Please use 'vut.visualization' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export all the functions for backward compatibility
__all__ = [
    "make_video",
    "plot_action_segmentation", 
    "plot_feature",
    "plot_features",
    "plot_image",
    "plot_images",
    "plot_metrics",
    "plot_palette",
    "plot_roc_curve",
    "plot_scatter",
]
