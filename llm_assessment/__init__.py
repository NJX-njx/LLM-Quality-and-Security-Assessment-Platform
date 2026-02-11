"""
LLM Quality and Security Assessment Platform

A unified platform for evaluating LLM capabilities, security, and alignment.
"""

__version__ = "0.1.0"
__author__ = "LLM Assessment Team"

from .core.assessment import AssessmentPlatform
from .core.report import ReportGenerator

__all__ = ["AssessmentPlatform", "ReportGenerator"]
