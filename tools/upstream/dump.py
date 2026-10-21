"""Dump upstream data that the TS port copies verbatim.

Run with the Python that has lightrag-hku installed (isolated mode):
    <venv>/bin/python -I tools/upstream/dump.py tools/upstream/upstream.json

Output: {"version", "prompts", "html5_entities", "constants"} as JSON.
Then `node tools/upstream/gen.ts` writes src/prompt.ts and src/internal/html5_entities.ts.
"""

import html.entities
import json
import sys

import lightrag
from lightrag import constants
from lightrag.prompt import PROMPTS

out = {
    "version": lightrag.__version__,
    "prompts": PROMPTS,
    "html5_entities": html.entities.html5,
    # private tables of CPython's html.unescape, needed for byte-identical output
    "html_invalid_charrefs": {str(k): v for k, v in html._invalid_charrefs.items()},
    "html_invalid_codepoints": sorted(html._invalid_codepoints),
    "python": sys.version,
    "constants": {
        k: getattr(constants, k)
        for k in dir(constants)
        if k.isupper() and isinstance(getattr(constants, k), (str, int, float, bool, list))
    },
}

with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
