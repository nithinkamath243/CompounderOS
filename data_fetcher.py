from typing import Dict, Optional

try:
    import requests
except Exception:
    requests = None

import urllib.error
import urllib.request


def http_get_text(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 20):
    try:
        if requests is not None:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code != 200:
                return None
            return response.text
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError, Exception):
        return None
