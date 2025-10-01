
from urllib.parse import urlsplit, unquote
from collections import Counter, defaultdict

def _normalize_host(netloc: str) -> str:
    """
    Lowercase host and strip an explicit port if present.
    (If you want to keep ports, remove the split(':', 1).)
    """
    host = netloc.lower()
    if '@' in host:  # strip userinfo if any
        host = host.split('@', 1)[1]
    host = host.split(':', 1)[0]
    if host.startswith('www.'):  # optional: collapse "www."
        host = host[4:]
    return host

def _path_segments(path: str) -> list[str]:
    """
    Split path into clean, decoded segments (no leading/trailing empties).
    """
    segs = [unquote(s) for s in path.split('/') if s]  # drop empty segments
    return segs

def truncate_uri_to_namespace(uri: str) -> str:
    """
    Turn a URI into its namespace key:
      - For URIs *with* a fragment: keep all path segments (ignore the fragment).
      - For URIs *without* a fragment: keep path up to the second-last segment.
    Examples:
      http://ex.org/a/b/c        -> ex.org/a/b
      http://ex.org/a/b/c#frag   -> ex.org/a/b/c
      http://ex.org/a/           -> ex.org/a
      http://ex.org/             -> ex.org
    """
    parts = urlsplit(uri)
    host = _normalize_host(parts.netloc)
    scheme = parts.scheme.lower()
    segs = _path_segments(parts.path)

    delimiter = "/"

    if parts.fragment:
        keep = segs  # keep full path when there's a fragment
        delimiter = "#"
    else:
        keep = segs[:-1] if segs else []

    if keep:
        return scheme + "://" + host + "/" + "/".join(keep) + delimiter
    return scheme + "://" + host + delimiter

# ---------- Counting-only API ----------

def count_namespace_usage(uris: list[str]) -> Counter:
    """
    Returns a Counter mapping truncated-namespace -> count.
    """
    c = Counter()
    for u in uris:
        try:
            key = truncate_uri_to_namespace(u)
            if key:
                c[key] += 1
        except Exception:
            # Skip clearly malformed URIs; customize if you prefer to raise
            continue
    return c
