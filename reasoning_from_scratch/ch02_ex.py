# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

# Módulo de compatibilidade: as funções de geração com streaming foram movidas para ch02.py.
from .ch02 import (
    generate_text_basic_stream,
    generate_text_basic_stream_cache,
)

__all__ = [
    "generate_text_basic_stream",
    "generate_text_basic_stream_cache",
]
