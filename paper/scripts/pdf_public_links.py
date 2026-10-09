"""Convert PDF file:// annotations to pinned public repository links."""
from pathlib import Path
from urllib.parse import unquote, urlparse
import subprocess
from pypdf.generic import NameObject, TextStringObject

def pin_public_links(writer, root):
    root = root.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    prefix = "https://github.com/RockLi/brian2-atlas-preprint/blob/" + commit + "/"
    for page in writer.pages:
        for ref in page.get("/Annots", []):
            annotation = ref.get_object()
            action = annotation.get("/A", {})
            uri = str(action.get("/URI", ""))
            if not uri.startswith("file:"):
                continue
            parsed = urlparse(uri)
            try:
                relative = Path(unquote(parsed.path)).resolve().relative_to(root)
            except ValueError:
                del annotation["/A"]
                continue
            action[NameObject("/URI")] = TextStringObject(prefix + relative.as_posix() + ("#" + parsed.fragment if parsed.fragment else ""))
