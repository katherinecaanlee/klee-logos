import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a local .env file if present.
load_dotenv()

# Base directory for backend assets.
BASE_DIR = Path(__file__).resolve().parent

# Path to the PDF design guide (user noted it lives in backend/).
PDF_PATH = Path(os.getenv("LOGO_GUIDE_PATH", BASE_DIR / "Prompt Engineering for AI Image Generation.pdf"))

# Where generated logo files will be written.
OUTPUT_DIR = Path(os.getenv("LOGO_OUTPUT_DIR", BASE_DIR / "generated_logos"))
OUTPUT_PREFIX = os.getenv("LOGO_OUTPUT_PREFIX", "logo")
STATIC_URL_PATH = os.getenv("LOGO_STATIC_URL_PATH", "/logos")

# Model choices can be overridden via environment variables if desired.
MODEL_A = os.getenv("LOGO_MODEL_A", "gpt-4.1")
MODEL_A_JUSTIFY = os.getenv("LOGO_MODEL_A_JUSTIFY", "gpt-4.1-mini")
MODEL_B = os.getenv("LOGO_MODEL_B", "gpt-image-1")

# Image generation defaults.
IMAGE_SIZE = os.getenv("LOGO_IMAGE_SIZE", "1024x1024")
IMAGE_BACKGROUND = os.getenv("LOGO_IMAGE_BACKGROUND", "transparent")

# Allow word marks by default unless explicitly disabled.
DEFAULT_ALLOW_WORD_MARK = os.getenv("LOGO_ALLOW_WORD_MARK", "true").lower() == "true"


def ensure_output_dir() -> None:
    """Create the output directory if it does not already exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
