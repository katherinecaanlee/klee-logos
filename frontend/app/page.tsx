/* eslint-disable @next/next/no-img-element */
// Using <img> keeps data URLs and arbitrary backend hosts working without extra Next image config.
"use client";

import { useState } from "react";

type LogoResult = {
  prompt: string;
  justification: string;
  image_path: string;
  image_url: string;
  image_base64: string;
};

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000").replace(
  /\/$/,
  "",
);

function formatImageSrc(logo: LogoResult): string | null {
  if (logo.image_url) {
    const isAbsolute = /^https?:\/\//.test(logo.image_url);
    if (isAbsolute) return logo.image_url;
    const base = API_BASE_URL.endsWith("/") ? API_BASE_URL.slice(0, -1) : API_BASE_URL;
    const path = logo.image_url.startsWith("/") ? logo.image_url : `/${logo.image_url}`;
    return `${base}${path}`;
  }
  if (logo.image_base64) {
    return `data:image/png;base64,${logo.image_base64}`;
  }
  return null;
}

export default function Home() {
  const [transcript, setTranscript] = useState("");
  const [logos, setLogos] = useState<LogoResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastRunSummary, setLastRunSummary] = useState<string | null>(null);

  const endpoint = `${API_BASE_URL}/generate`;

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmedTranscript = transcript.trim();

    if (!trimmedTranscript) {
      setError("Please paste a transcript before generating.");
      return;
    }

    setLoading(true);
    setError(null);
    setLogos([]);

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          transcript: trimmedTranscript,
        }),
      });

      const payload = await response.json();

      if (!response.ok) {
        const message = payload?.detail || "The logo service returned an error.";
        throw new Error(message);
      }

      setLogos(payload?.logos ?? []);
      setLastRunSummary(`${trimmedTranscript.slice(0, 120)}${trimmedTranscript.length > 120 ? "…" : ""}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unexpected error while generating logos.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const hasResults = logos.length > 0;

  return (
    <div className="min-h-screen px-5 py-8 md:px-10 md:py-12">
      <div className="mx-auto flex max-w-6xl flex-col gap-8">
        <div className="rounded-3xl border border-[#e3e7ef] bg-white/80 p-8 shadow-[0_24px_80px_rgba(15,23,42,0.08)] backdrop-blur-xl md:p-10">
          <div className="grid gap-10 lg:grid-cols-[1.05fr_0.95fr] lg:items-start">
            <div className="space-y-4">
              <h1 className="text-3xl font-semibold tracking-tight text-slate-900 sm:text-4xl">
                Logo Generator
              </h1>
              
            </div>

            <form onSubmit={handleSubmit} className="space-y-5">
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <label htmlFor="transcript" className="font-medium text-slate-900">
                    Transcript
                  </label>
                  <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] uppercase tracking-wide text-slate-500">
                    Required
                  </span>
                </div>
                <div className="rounded-2xl border border-slate-200 bg-slate-50/70 shadow-inner shadow-slate-200/70 transition focus-within:border-slate-400 focus-within:shadow-[0_12px_38px_rgba(15,23,42,0.08)]">
                  <textarea
                    id="transcript"
                    value={transcript}
                    onChange={(event) => setTranscript(event.target.value)}
                    rows={7}
                    placeholder="Paste the creative brief or transcript that describes the brand, tone, and constraints."
                    className="w-full resize-none border-0 bg-transparent px-4 py-3 text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none"
                  />
                </div>
              </div>

              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                
                <button
                  type="submit"
                  disabled={loading}
                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-slate-900 to-slate-700 px-5 py-3 text-sm font-semibold text-white shadow-[0_14px_40px_rgba(15,23,42,0.25)] transition hover:-translate-y-[1px] hover:shadow-[0_18px_50px_rgba(15,23,42,0.28)] disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {loading ? "Generating..." : "Generate logos"}
                </button>
              </div>
            </form>
          </div>
        </div>

        <div className="space-y-4">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-xs uppercase tracking-wide text-slate-500">Output review</p>
              <h2 className="text-xl font-semibold text-slate-900">
                {hasResults ? "3 concepts ready" : "Awaiting a run"}
              </h2>
            </div>

          </div>

          {error && (
            <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800 shadow-[0_12px_30px_rgba(225,29,72,0.12)]">
              {error}
            </div>
          )}

          {!hasResults && !loading && (
            <div className="rounded-3xl border border-dashed border-slate-200 bg-white/70 p-10 text-center text-slate-600 shadow-[0_18px_48px_rgba(15,23,42,0.06)]">
              <p className="text-lg font-medium text-slate-900">Run the generator to see options</p>

            </div>
          )}

          {hasResults && (
            <div className="grid gap-4 md:grid-cols-2">
              {logos.map((logo, index) => {
                const imageSrc = formatImageSrc(logo);
                return (
                  <div
                    key={logo.image_path || logo.prompt || index}
                    className="group flex flex-col gap-4 rounded-3xl border border-[#e3e7ef] bg-white p-4 shadow-[0_18px_50px_rgba(15,23,42,0.08)] transition hover:-translate-y-[2px] hover:shadow-[0_22px_60px_rgba(15,23,42,0.12)]"
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex h-28 w-28 items-center justify-center overflow-hidden rounded-2xl border border-slate-200 bg-slate-50 shadow-inner shadow-slate-200/70">
                        {imageSrc ? (
                          <img
                            src={imageSrc}
                            alt={`Logo concept ${index + 1}`}
                            className="h-full w-full object-cover"
                          />
                        ) : (
                          <span className="text-xs text-slate-500">Image unavailable</span>
                        )}
                      </div>
                      <div className="flex-1 space-y-2">
                        <div className="flex flex-wrap items-center gap-2">
                        </div>
                        <p className="text-sm text-slate-700">{logo.justification}</p>
                      </div>
                    </div>
                    
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
