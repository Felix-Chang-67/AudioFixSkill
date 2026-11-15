#!/usr/bin/env python3
"""Entry point: python3 scripts/audiofix.py <check|profile|analyze|restore|verify|auto> ...  (see SKILL.md)"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audiofix.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
