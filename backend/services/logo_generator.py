import base64
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from ..config import (
    DEFAULT_ALLOW_WORD_MARK,
    IMAGE_BACKGROUND,
    IMAGE_SIZE,
    MODEL_A,
    MODEL_A_JUSTIFY,
    MODEL_B,
    OUTPUT_DIR,
    OUTPUT_PREFIX,
    PDF_PATH,
    STATIC_URL_PATH,
    ensure_output_dir,
)
from ..schemas import LogoResult

# Ensure environment variables (e.g., OPENAI_API_KEY) are loaded.
load_dotenv()

# Shared render constraints appended to every image prompt.
IMAGE_RENDER_CONSTRAINTS = (
    " Render a bold, clearly visible logo mark centered in frame, high contrast, "
    "thick enough strokes to be visible. Do not output an empty or nearly blank canvas."
)


def build_logo_type_text(allow_word_mark: bool) -> Tuple[str, str]:
    """Return allowed logo options text and guidance for word marks."""
    base_options = ["- pictorial mark", "- abstract mark"]
    word_mark_guidance = (
        "pictorial_logo: symbolic, simplified (NO WORDS ALLOWED).\n"
        "abstract_icon: geometric, non-literal, no identifiable object, MUST BE SYMMETRIC (NO WORDS ALLOWED).\n"
    )

    if allow_word_mark:
        base_options.append("- word mark")
        word_mark_guidance += (
            "word_mark: must feel like custom typography - not generic default text. "
            "Describe typographic character (style, proportions, spacing/rhythm) so the image model "
            "invents a subtle, creative, coherent structural nuance without you naming a specific trick. "
            "Encourage it to be clever.\n"
        )
    else:
        word_mark_guidance += "Word marks are disabled for this request; do not include text in any logo.\n"

    return "\n".join(base_options), word_mark_guidance


def extract_prompts(raw_text: str) -> List[str]:
    """Extract PROMPT_1/2/3 blocks from the model output."""
    def _extract(label: str, text: str) -> str:
        idx = text.find(label)
        if idx == -1:
            return ""
        start = idx + len(label)
        next_labels = ["PROMPT_1:", "PROMPT_2:", "PROMPT_3:"]
        next_idx = None
        for nl in next_labels:
            if nl == label:
                continue
            j = text.find(nl, start)
            if j != -1:
                next_idx = j if next_idx is None else min(next_idx, j)
        chunk = text[start:next_idx].strip() if next_idx else text[start:].strip()
        return chunk

    prompts = [
        _extract("PROMPT_1:", raw_text),
        _extract("PROMPT_2:", raw_text),
        _extract("PROMPT_3:", raw_text),
    ]
    if not all(prompts):
        raise RuntimeError(
            "Failed to parse prompts from model output. Ensure Model A returns PROMPT_1/2/3 blocks."
        )
    return prompts


class LogoGenerator:
    """Encapsulates the multi-step logo generation flow."""

    def __init__(self, client: OpenAI | None = None):
        self.client = client or OpenAI()

    def generate(
        self,
        transcript: str,
        brand_name: str = "",
        allow_word_mark: bool | None = None,
    ) -> List[LogoResult]:
        allow_word_mark = DEFAULT_ALLOW_WORD_MARK if allow_word_mark is None else allow_word_mark
        ensure_output_dir()

        pdf_file_id = self._upload_pdf_guide()
        logo_type_text, word_mark_guidance = build_logo_type_text(allow_word_mark)
        prompts = self._generate_prompts(
            pdf_file_id=pdf_file_id,
            transcript=transcript,
            brand_name=brand_name,
            logo_type_options_text=logo_type_text,
            word_mark_guidance=word_mark_guidance,
        )

        run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        results: List[LogoResult] = []

        for idx, prompt in enumerate(prompts, start=1):
            final_prompt = prompt + IMAGE_RENDER_CONSTRAINTS
            image_base64 = self._generate_image(final_prompt)
            image_path = self._save_image(image_base64, run_id=run_id, index=idx)
            logo_file_id = self._upload_logo_image(image_path)
            justification = self._justify_logo(
                pdf_file_id=pdf_file_id,
                logo_file_id=logo_file_id,
                transcript=transcript,
                brand_name=brand_name,
                final_prompt=final_prompt,
            )
            results.append(
                LogoResult(
                    prompt=prompt,
                    justification=justification,
                    image_path=str(image_path),
                    image_url=f"{STATIC_URL_PATH}/{image_path.name}",
                    image_base64=image_base64,
                )
            )

        return results

    def _upload_pdf_guide(self) -> str:
        if not PDF_PATH.exists():
            raise FileNotFoundError(
                f"Design guide PDF not found at {PDF_PATH}. Ensure the file exists or set LOGO_GUIDE_PATH."
            )

        with open(PDF_PATH, "rb") as f:
            pdf_file = self.client.files.create(file=f, purpose="user_data")
        return pdf_file.id

    def _generate_prompts(
        self,
        pdf_file_id: str,
        transcript: str,
        brand_name: str,
        logo_type_options_text: str,
        word_mark_guidance: str,
    ) -> List[str]:
        response = self.client.responses.create(
            model=MODEL_A,
            instructions=(
                "You are Model A, a prompt engineer for a modernist logo generator. "
                "Your job is to output THREE distinct logo-generation prompts for an image model. "
                "If the transcript says not to do something, do not do it. "
                "All three must stay faithful to the brand transcript, but differ meaningfully via "
                "controlled design lenses (not random style changes). "
                "Each prompt must specify a logo type chosen ONLY from the allowed options. "
                "Output must be EXACTLY three prompts, each on its own block, labeled:\n"
                "PROMPT_1:\nPROMPT_2:\nPROMPT_3:\n"
                "No other text."
            ),
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_id": pdf_file_id},
                        {
                            "type": "input_text",
                            "text": f"""
You are creating prompts for an AI image model (Model B) that designs modernist logos.

Allowed logo types (choose one per concept, but you do NOT need to use all types):
{logo_type_options_text}
If a transcript indicates NOT ABSTRACT: Go with pictorial or word mark.

Brand owner transcript:
\"\"\"{transcript}\"\"\"

Brand name (optional; may also appear in transcript):
\"\"\"{brand_name}\"\"\"

Guidance on text usage:
{word_mark_guidance}

Task:
Generate THREE distinct logo prompts for the same client. They must feel like three
professional options a design studio would present - different enough to compare,
but not crazy or off-brief.

All three prompts must share these constants:
- modernist, minimal, balanced composition
- should NOT be cluttered
- should NOT be too complex
- monochrome, vector-like, strong legibility at small sizes
- reflect the transcript's personality and constraints
- do not do prompt anything the transcript does not want
- encourage a hint of flair and creativity 

Make them distinct by using three different concept lenses:
1) STRUCTURAL / STABLE: symmetry, grounded geometry, calm authority
2) DIRECTIONAL / PROGRESSIVE: subtle asymmetry or forward cue, still restrained
3) REDUCTIVE / PREMIUM: fewer elements, more negative space, quiet confidence

For each concept:
- Choose the most appropriate logo type from the allowed list.
- Describe the mark precisely (geometry, proportion, negative space strategy, typographic character if word mark).
- If the chosen type disallows words, explicitly ensure no text is present in the logo.

Output format (required):
PROMPT_1: <single-block prompt text>
PROMPT_2: <single-block prompt text>
PROMPT_3: <single-block prompt text>

No explanations.
""",
                        },
                    ],
                }
            ],
        )

        raw = response.output_text.strip()
        return extract_prompts(raw)

    def _generate_image(self, prompt: str) -> str:
        image_response = self.client.images.generate(
            model=MODEL_B,
            prompt=prompt,
            size=IMAGE_SIZE,
            background=IMAGE_BACKGROUND,
            output_format="png",
        )
        return image_response.data[0].b64_json

    def _save_image(self, image_base64: str, run_id: str, index: int) -> Path:
        filename = f"{OUTPUT_PREFIX}_{run_id}_{index}.png"
        path = OUTPUT_DIR / filename
        with open(path, "wb") as f:
            f.write(base64.b64decode(image_base64))
        return path

    def _upload_logo_image(self, image_path: Path) -> str:
        with open(image_path, "rb") as img_file:
            logo_file = self.client.files.create(file=img_file, purpose="vision")
        return logo_file.id

    def _justify_logo(
        self,
        pdf_file_id: str,
        logo_file_id: str,
        transcript: str,
        brand_name: str,
        final_prompt: str,
    ) -> str:
        response = self.client.responses.create(
            model=MODEL_A_JUSTIFY,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are Model A, the modernist logo designer and prompt engineer who created this logo.\n"
                                "You have access to the brand transcript, the design guide PDF, and the final image, plus the exact prompt "
                                "that was sent to the image model.\n\n"
                                "Write EXACTLY ONE very short concise not wordy sentence without too many, in first person plural ('we'), explaining the design rationale for this "
                                "specific logo: how its forms, geometry, and visual decisions interpret the brief.\n"
                                "Do not add bullet points, labels, or multiple sentences.\n\n"
                                f"BRAND TRANSCRIPT:\n{transcript}\n\n"
                                f"BRAND NAME (may be empty): {brand_name}\n\n"
                                f"IMAGE PROMPT USED:\n{final_prompt}\n"
                            ),
                        },
                        {"type": "input_file", "file_id": pdf_file_id},
                        {"type": "input_image", "file_id": logo_file_id, "detail": "auto"},
                    ],
                }
            ],
        )
        return response.output_text.strip().replace("\n", " ")
