import sys
import matplotlib
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


matplotlib.use("Agg")
