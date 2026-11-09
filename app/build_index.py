"""Build the search index (run once): python build_index.py [chunks.json]

With no argument it uses helpdesk.ingest (when it exists), otherwise the temporary dev/chunks.json.
"""
import json
import sys
from pathlib import Path

from helpdesk.search import INDEX_DIR, build_index

ROOT = Path(__file__).resolve().parent


def load_chunks():
    if len(sys.argv) > 1:
        return json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        from helpdesk import ingest                       # teammate's module, when it is ready
        return ingest.build_chunks()
    except ImportError:
        dev = ROOT / "dev" / "chunks.json"
        if not dev.exists():
            sys.exit("No helpdesk/ingest.py yet. Make test chunks first: python dev/dev_chunks.py")
        print("helpdesk/ingest.py not found: using the temporary dev/chunks.json")
        return json.loads(dev.read_text(encoding="utf-8"))


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"indexed {build_index(chunks)} chunks into {INDEX_DIR}")
