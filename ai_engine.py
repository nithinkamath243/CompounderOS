import logging
import random
import time
from typing import Iterable, List, Optional, Tuple

from google import genai

from .config import get_service_key


class GeminiEngine:
    # Config-headed preference ladder. GEMINI_MODEL (config) is consulted at
    # runtime in model_candidates(); these are the static fallbacks, newest first.
    DEFAULT_MODELS = [
        "gemini-3.5-flash",
        "gemini-3-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._configured_key: Optional[str] = None
        self._models: Optional[List[str]] = None
        self.last_error: Optional[str] = None
        self._last_empty_reason: Optional[str] = None
        self._logger = logging.getLogger(__name__)
        self.client = None

    def set_api_key(self, api_key: Optional[str]) -> None:
        if api_key == self.api_key:
            return
        self.api_key = api_key
        self._configured_key = None
        self._models = None
        self.last_error = None
        self.client = None

    def _configure(self) -> bool:
        if not self.api_key:
            return False
        if self._configured_key == self.api_key and self.client:
            return True
        try:
            self.client = genai.Client(api_key=self.api_key)
            self._configured_key = self.api_key
            return True
        except Exception as e:
            self._logger.warning("Gemini configure failed: %s", e)
            self.client = None
            return False

    def list_models(self, refresh: bool = False) -> List[str]:
        if not self._configure():
            return []
        if self._models is not None and not refresh:
            return self._models
        try:
            self._models = []
            for m in self.client.models.list():
                name = m.name.replace("models/", "")
                self._models.append(name)
        except Exception as e:
            self._logger.warning("Gemini list_models failed: %s", e)
            self._models = []
        return self._models

    def model_candidates(self, preferred: Optional[Iterable[str]] = None) -> List[str]:
        preferred_models = list(preferred) if preferred else list(self.DEFAULT_MODELS)
        configured = get_service_key("GEMINI_MODEL")
        if configured:
            preferred_models = [configured, *preferred_models]
        preferred_models = [m.replace("models/", "") for m in preferred_models]

        discovered = self.list_models(refresh=False)
        banned_tokens = (
            "tts",
            "image",
            "embedding",
            "aqa",
            "veo",
            "live",
            "speech",
        )

        def usable(model: str) -> bool:
            if not model:
                return False
            low = model.lower()
            if any(tok in low for tok in banned_tokens):
                return False
            if not ("flash" in low or "pro" in low):
                return False
            return True

        out: List[str] = []
        seen = set()
        for model in [*preferred_models, *discovered]:
            if not usable(model):
                continue
            if model not in seen:
                out.append(model)
                seen.add(model)
        return out[:8]

    def _extract_text(self, response) -> Optional[str]:
        """Robust text extraction for the google-genai SDK.

        response.text can be None (or raise) when candidates carry no simple
        text parts — safety blocks, MAX_TOKENS with empty content, tool parts.
        Walk the parts manually and record WHY extraction failed so the UI can
        surface a real reason instead of a generic 'empty response'.
        """
        self._last_empty_reason = None
        text = None
        try:
            text = getattr(response, "text", None)
        except Exception:
            text = None
        if text:
            return text
        try:
            candidates = getattr(response, "candidates", None) or []
            for cand in candidates:
                content = getattr(cand, "content", None)
                parts = getattr(content, "parts", None) or []
                chunks = [getattr(p, "text", None) for p in parts]
                chunks = [c for c in chunks if c]
                if chunks:
                    return "".join(chunks)
            reason_bits = []
            pf = getattr(response, "prompt_feedback", None)
            block = getattr(pf, "block_reason", None) if pf is not None else None
            if block:
                reason_bits.append(f"prompt blocked: {block}")
            if candidates:
                fr = getattr(candidates[0], "finish_reason", None)
                if fr:
                    reason_bits.append(f"finish_reason: {fr}")
            self._last_empty_reason = "; ".join(reason_bits) if reason_bits else "no text parts in response"
        except Exception as e:
            self._last_empty_reason = f"response parse error: {e}"
        return None

    def generate(self, prompt: str, preferred: Optional[Iterable[str]] = None) -> Tuple[Optional[str], Optional[str]]:
        if not self._configure():
            self.last_error = "Gemini API key missing or invalid"
            return None, None

        self.last_error = None
        last_error_msg: Optional[str] = None

        for model_name in self.model_candidates(preferred=preferred):
            backoff = 0.6
            for attempt in range(3):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    text = self._extract_text(response)
                    if text:
                        self.last_error = None
                        return text, model_name
                    detail = f" ({self._last_empty_reason})" if self._last_empty_reason else ""
                    last_error_msg = f"Empty response from {model_name}{detail}"
                    break
                except Exception as e:
                    msg = str(e)
                    low = msg.lower()
                    retryable = any(
                        token in low for token in ("429", "rate limit", "quota", "resource exhausted", "temporarily unavailable")
                    )
                    if retryable and attempt < 2:
                        sleep_s = min(4.0, backoff * (2 ** attempt) + random.uniform(0, 0.25))
                        time.sleep(sleep_s)
                        continue
                    self._logger.warning("Gemini generate failed for %s: %s", model_name, e)
                    last_error_msg = f"{model_name}: {msg}"
                    break

        self.last_error = last_error_msg or "No Gemini model could generate content"
        return None, None
