#!/usr/bin/env python3
"""
Compatibility shim for the expected entrypoint name `transcript_to_logos.py`.
It reuses the CLI logic defined in transcript_to_svg.py.
"""

from transcript_to_svg import cli_main


if __name__ == "__main__":
    cli_main()
