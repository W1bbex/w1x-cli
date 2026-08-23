#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""W1x CLI — Fast & Intelligent Terminal AI Client for Ollama.

Run directly with: python main.py
"""
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from w1x.app import main

if __name__ == "__main__":
    main()
