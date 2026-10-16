"""Model files' names, of which models are found, with no engine to import: a GGUF split in
parts, as llama.cpp's gguf-split names them, and a vision projector, as its converters name one."""

import re
from pathlib import Path

# a split GGUF's part: its prefix, its number from 1, and their count
PART = re.compile(r"(.+)-(\d{5})-of-(\d{5})\.gguf")


def model_name(path: Path | str) -> str:
    """A model's name, as its file's stem, of a split GGUF's part without its numbers."""
    name = Path(path).name
    named = PART.fullmatch(name)
    return named[1] if named else Path(name).stem


def later_part(path: Path | str) -> bool:
    """Whether a file is a part of a split GGUF after its first, which a model is not of its own."""
    named = PART.fullmatch(Path(path).name)
    return named is not None and int(named[2]) > 1


def projector(path: Path | str) -> bool:
    """Whether a GGUF is a vision projector, as llama.cpp's converters name them: mmproj-*.gguf."""
    return "mmproj" in Path(path).name.lower()
