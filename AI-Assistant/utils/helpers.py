from pathlib import Path
from typing import List
import logging

# logging.basicConfig(level=logging.INFO)  # Removed central logging config
logger = logging.getLogger(__name__)

def format_sources(sources: List[dict]) -> str:
    """Format source information for display."""
    if not sources:
        return "No sources found."
    
    formatted = "\n\nSources:\n"
    for i, source in enumerate(sources, 1):
        formatted += f"\n{i}. "
        if 'source' in source['metadata']:
            formatted += f"File: {Path(source['metadata']['source']).name}\n"
        if 'page' in source['metadata']:
            formatted += f"   Page: {source['metadata']['page']}\n"
        # formatted += f"   Preview: {source['content']}\n"
    
    return formatted


def fix_arabic_text(text: str) -> str:
    """
    Fixes Arabic text for terminal display by reshaping and reordering (RTL).
    """
    if not text:
        return text
        
    # Detect if there's any Arabic character
    has_arabic = any('\u0600' <= char <= '\u06FF' for char in text)
    if not has_arabic:
        return text
        
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        
        # Reshape Arabic characters (join them)
        reshaped_text = arabic_reshaper.reshape(text)
        
        # Apply Bidi algorithm for RTL display
        bidi_text = get_display(reshaped_text)
        
        return bidi_text
    except ImportError:
        # Fallback if libraries are not installed
        return text


def print_divider(char: str = "-", length: int = 50):
    """Print a divider line."""
    print(char * length)