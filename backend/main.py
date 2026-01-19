from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAIError

from .config import OUTPUT_DIR, STATIC_URL_PATH, ensure_output_dir
from .schemas import GenerateLogosRequest, GenerateLogosResponse
from .services.logo_generator import LogoGenerator

# Ensure output directory exists before mounting static files.
ensure_output_dir()

app = FastAPI(title="Logo Generator API", version="1.0.0")

# Basic CORS to allow calls from a separate Next.js front end.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated logo files so the front end can fetch them by URL.
app.mount(STATIC_URL_PATH, StaticFiles(directory=OUTPUT_DIR), name="logos")

logo_generator = LogoGenerator()


@app.post("/generate", response_model=GenerateLogosResponse)
async def generate_logos(payload: GenerateLogosRequest) -> GenerateLogosResponse:
    try:
        logos = await run_in_threadpool(
            logo_generator.generate,
            payload.transcript,
            payload.brand_name or "",
            payload.allow_word_mark,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unexpected error during logo generation") from exc

    return GenerateLogosResponse(logos=logos)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
