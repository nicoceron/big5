# Expose the DExpertGenerator implementation
try:
    from .dexpert import DExpertGenerator
except ImportError:
    pass

# Try to expose the local Ollama-based implementation if available
try:
    from .dexpert_local import DExpertGenerator as DExpertGeneratorLocal
except ImportError:
    pass
