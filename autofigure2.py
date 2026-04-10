"""
Paper Method to SVG icon replacement full pipeline (Label mode enhanced + Box merge + multi-prompt support)

Supported API Providers:
- openrouter: OpenRouter API (https://openrouter.ai/api/v1)
- bianxie: Bianxie API (https://api.bianxie.ai/v1) - uses OpenAI SDK
- gemini: Google Gemini official API (https://ai.google.dev/)

Placeholder mode (--placeholder_mode):
- none: no special style (default black border)
- box: pass boxlib coordinates to LLM
- label: gray fill+black border+numbered labels <AF>01, <AF>02... (recommended)

SAM3 multi-prompt support (--sam_prompt):
- supports multiple comma-separated text prompts
- example: "icon,diagram,arrow,chart"
- detects separately for each prompt, then merges and deduplicates results
- boxlib.json records the source prompt for each box

Box merge feature (--merge_threshold):
- merges and deduplicates overlapping boxes from SAM3 detection
- overlap ratio = intersection area / smaller box area
- default threshold 0.9, set to 0 to disable merging
- cross-prompt detection results are also automatically deduplicated

Pipeline:
1. Input paper method text, call LLM to generate academic-style image -> figure.png
2. SAM3 segments image with gray fill+black border+numbered labels -> samed.png + boxlib.json
   2.1 Supports multiple text prompts detected separately
   2.2 Merges overlapping boxes (optional, controlled by --merge_threshold)
3. Crop and remove background with RMBG2 -> icons/icon_AF01_nobg.png, icon_AF02_nobg.png...
4. Multimodal call to generate SVG (placeholder style matches samed.png) -> template.svg
4.5. SVG syntax validation (lxml) + LLM repair
4.6. LLM optimizes SVG template (position and style alignment) -> optimized_template.svg
     Number of iterations controlled by --optimize_iterations (0 = skip optimization)
4.7. Coordinate alignment: compare figure.png and SVG dimensions, compute scale factors
5. Replace transparent icons into SVG placeholders by label matching -> final.svg

Usage:
    # Use Bianxie + label mode (default)
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --api_key "your-key"

    # Use OpenRouter
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --api_key "sk-or-v1-xxx" --provider openrouter

    # Use box mode (pass coordinates)
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --placeholder_mode box

    # Use multiple SAM3 prompts for detection
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --sam_prompt "icon,diagram,arrow"

    # Skip step 4.6 optimization (set iterations to 0)
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --optimize_iterations 0

    # Set step 4.6 to optimize for 3 iterations
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --optimize_iterations 3

    # Custom box merge threshold (0.8)
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --merge_threshold 0.8

    # Disable box merging
    python iou_autofigure.py --method_file paper_method.txt --output_dir ./output --merge_threshold 0
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

import requests
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from transformers import AutoModelForImageSegmentation


# ============================================================================
# Provider configuration
# ============================================================================

PROVIDER_CONFIGS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_image_model": "google/gemini-3-pro-image-preview",
        "default_svg_model": "google/gemini-3.1-pro-preview",
    },
    "bianxie": {
        "base_url": "https://api.bianxie.ai/v1",
        "default_image_model": "gemini-3-pro-image-preview",
        "default_svg_model": "gemini-3.1-pro-preview",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "default_image_model": "gemini-3-pro-image-preview",
        "default_svg_model": "gemini-3.1-pro",
    },
}

ProviderType = Literal["openrouter", "bianxie", "gemini"]
PlaceholderMode = Literal["none", "box", "label"]
GEMINI_DEFAULT_IMAGE_SIZE = "4K"
IMAGE_SIZE_CHOICES = ("1K", "2K", "4K")
BOXLIB_NO_ICON_MODE_KEY = "no_icon_mode"

# SAM3 API config
SAM3_FAL_API_URL = "https://fal.run/fal-ai/sam-3/image"
SAM3_ROBOFLOW_API_URL = os.environ.get(
    "ROBOFLOW_API_URL",
    "https://serverless.roboflow.com/sam3/concept_segment",
)
SAM3_API_TIMEOUT = 300

# Step 1 reference image settings (overridden by CLI)
USE_REFERENCE_IMAGE = False
REFERENCE_IMAGE_PATH: Optional[str] = None


# ============================================================================
# Unified LLM call interface
# ============================================================================

def call_llm_text(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """
    Unified text LLM call interface

    Args:
        prompt: text prompt
        api_key: API key
        model: model name
        base_url: API base URL
        provider: API provider
        reference_image: reference image (optional)
        max_tokens: max output token count
        temperature: temperature parameter

    Returns:
        LLM response text
    """
    if provider == "bianxie":
        return _call_bianxie_text(prompt, api_key, model, base_url, max_tokens, temperature)
    if provider == "gemini":
        return _call_gemini_text(prompt, api_key, model, max_tokens, temperature)
    return _call_openrouter_text(prompt, api_key, model, base_url, max_tokens, temperature)


def call_llm_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """
    Unified multimodal LLM call interface

    Args:
        contents: content list (strings or PIL Images)
        api_key: API key
        model: model name
        base_url: API base URL
        provider: API provider
        max_tokens: max output token count
        temperature: temperature parameter

    Returns:
        LLM response text
    """
    if provider == "bianxie":
        return _call_bianxie_multimodal(contents, api_key, model, base_url, max_tokens, temperature)
    if provider == "gemini":
        return _call_gemini_multimodal(contents, api_key, model, max_tokens, temperature)
    return _call_openrouter_multimodal(contents, api_key, model, base_url, max_tokens, temperature)


def call_llm_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    reference_image: Optional[Image.Image] = None,
    image_size: str = GEMINI_DEFAULT_IMAGE_SIZE,
) -> Optional[Image.Image]:
    """
    Unified image generation LLM call interface

    Args:
        prompt: text prompt
        api_key: API key
        model: model name
        base_url: API base URL
        provider: API provider

    Returns:
        Generated PIL Image, or None on failure
    """
    if provider == "bianxie":
        return _call_bianxie_image_generation(prompt, api_key, model, base_url, reference_image)
    if provider == "gemini":
        return _call_gemini_image_generation(
            prompt=prompt,
            api_key=api_key,
            model=model,
            reference_image=reference_image,
            image_size=image_size,
        )
    return _call_openrouter_image_generation(prompt, api_key, model, base_url, reference_image)


# ============================================================================
# Bianxie provider implementation (using OpenAI SDK)
# ============================================================================

def _call_bianxie_text(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """Call Bianxie text API using OpenAI SDK"""
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)

        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return completion.choices[0].message.content if completion and completion.choices else None
    except Exception as e:
        print(f"[Bianxie] API call failed: {e}")
        raise


def _call_bianxie_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """Call Bianxie multimodal API using OpenAI SDK"""
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)

        message_content: List[Dict[str, Any]] = []
        for part in contents:
            if isinstance(part, str):
                message_content.append({"type": "text", "text": part})
            elif isinstance(part, Image.Image):
                buf = io.BytesIO()
                part.save(buf, format='PNG')
                image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                })

        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": message_content}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return completion.choices[0].message.content if completion and completion.choices else None
    except Exception as e:
        print(f"[Bianxie] Multimodal API call failed: {e}")
        raise


def _call_bianxie_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    reference_image: Optional[Image.Image] = None,
) -> Optional[Image.Image]:
    """Call Bianxie image generation API using OpenAI SDK"""
    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)

        if reference_image is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            buf = io.BytesIO()
            reference_image.save(buf, format='PNG')
            image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            message_content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ]
            messages = [{"role": "user", "content": message_content}]

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        content = completion.choices[0].message.content if completion and completion.choices else None

        if not content:
            return None

        # Bianxie returns image in Markdown format: ![text](data:image/png;base64,...)
        pattern = r'data:image/(png|jpeg|jpg|webp);base64,([A-Za-z0-9+/=]+)'
        match = re.search(pattern, content)

        if match:
            image_base64 = match.group(2)
            image_data = base64.b64decode(image_base64)
            return Image.open(io.BytesIO(image_data))

        return None
    except Exception as e:
        print(f"[Bianxie] Image generation API call failed: {e}")
        raise


# ============================================================================
# OpenRouter provider implementation (using requests)
# ============================================================================

def _get_openrouter_headers(api_key: str) -> dict:
    """Get OpenRouter request headers"""
    return {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'HTTP-Referer': 'https://localhost',
        'X-Title': 'MethodToSVG'
    }


def _get_openrouter_api_url(base_url: str) -> str:
    """Get OpenRouter API URL"""
    if not base_url.endswith('/chat/completions'):
        if base_url.endswith('/'):
            return base_url + 'chat/completions'
        else:
            return base_url + '/chat/completions'
    return base_url


def _extract_openrouter_message_text(message: Any) -> Optional[str]:
    """Extract text from OpenRouter message, compatible with string/list/object content shapes"""
    if not isinstance(message, dict):
        return None

    def _collect_from_part(part: Any, out: list[str]) -> None:
        if isinstance(part, str):
            text = part.strip()
            if text:
                out.append(text)
            return

        if not isinstance(part, dict):
            return

        for key in ("text", "content", "value"):
            value = part.get(key)
            if isinstance(value, str) and value.strip():
                out.append(value.strip())

        nested = part.get("content")
        if isinstance(nested, list):
            for item in nested:
                _collect_from_part(item, out)

    content = message.get("content")

    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, dict):
        chunks: list[str] = []
        _collect_from_part(content, chunks)
        if chunks:
            return "\n".join(chunks)

    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            _collect_from_part(part, chunks)
        if chunks:
            return "\n".join(chunks)

    for key in ("output_text", "text"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _summarize_openrouter_choice(choice: Any) -> str:
    """Build readable OpenRouter choice summary to help diagnose empty response issues"""
    if not isinstance(choice, dict):
        return f"invalid choice type={type(choice).__name__}"

    message = choice.get("message")
    if not isinstance(message, dict):
        return (
            f"finish_reason={choice.get('finish_reason')}, "
            f"message_type={type(message).__name__}"
        )

    content = message.get("content")
    content_type = type(content).__name__
    if isinstance(content, str):
        content_size = len(content)
    elif isinstance(content, list):
        content_size = len(content)
    elif isinstance(content, dict):
        content_size = len(content.keys())
    else:
        content_size = 0

    refusal = message.get("refusal")
    refusal_preview = repr(refusal)
    if len(refusal_preview) > 220:
        refusal_preview = refusal_preview[:220] + "..."

    return (
        f"finish_reason={choice.get('finish_reason')}, "
        f"message_keys={sorted(message.keys())}, "
        f"content_type={content_type}, "
        f"content_size={content_size}, "
        f"refusal={refusal_preview}"
    )


def _call_openrouter_text(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """Call OpenRouter text API using requests"""
    api_url = _get_openrouter_api_url(base_url)
    headers = _get_openrouter_headers(api_key)

    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens,
        'temperature': temperature,
        'stream': False
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=300)

    if response.status_code != 200:
        raise Exception(f'OpenRouter API error: {response.status_code} - {response.text[:500]}')

    result = response.json()

    if 'error' in result:
        error_msg = result.get('error', {})
        if isinstance(error_msg, dict):
            error_msg = error_msg.get('message', str(error_msg))
        raise Exception(f'OpenRouter API error: {error_msg}')

    choices = result.get('choices', [])
    if not choices:
        return None

    message = choices[0].get('message', {})
    text = _extract_openrouter_message_text(message)
    if text:
        return text
    return None


def _call_openrouter_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    base_url: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """Call OpenRouter multimodal API using requests"""
    api_url = _get_openrouter_api_url(base_url)
    headers = _get_openrouter_headers(api_key)

    message_content: List[Dict[str, Any]] = []
    for part in contents:
        if isinstance(part, str):
            message_content.append({"type": "text", "text": part})
        elif isinstance(part, Image.Image):
            buf = io.BytesIO()
            part.save(buf, format='PNG')
            image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"}
            })

    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': message_content}],
        'max_tokens': max_tokens,
        'temperature': temperature,
        'stream': False
    }

    retry_env = os.environ.get("OPENROUTER_MULTIMODAL_RETRIES", "3")
    delay_env = os.environ.get("OPENROUTER_MULTIMODAL_RETRY_DELAY", "1.5")
    try:
        retry_count = max(1, int(retry_env))
    except ValueError:
        retry_count = 3
    try:
        retry_delay = max(0.0, float(delay_env))
    except ValueError:
        retry_delay = 1.5

    last_error: Optional[Exception] = None
    for attempt in range(1, retry_count + 1):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=300)

            if response.status_code != 200:
                raise Exception(f'OpenRouter API error: {response.status_code} - {response.text[:500]}')

            result = response.json()

            if 'error' in result:
                error_msg = result.get('error', {})
                if isinstance(error_msg, dict):
                    error_msg = error_msg.get('message', str(error_msg))
                raise Exception(f'OpenRouter API error: {error_msg}')

            choices = result.get('choices', [])
            if not choices:
                raise RuntimeError("OpenRouter returned empty choices")

            message = choices[0].get('message', {})
            text = _extract_openrouter_message_text(message)
            if text:
                return text

            choice_summary = _summarize_openrouter_choice(choices[0])
            raise RuntimeError(
                "OpenRouter multimodal response has no parseable text content."
                f" model={model}, summary={choice_summary}"
            )
        except Exception as e:
            last_error = e
            if attempt < retry_count:
                sleep_s = retry_delay * (2 ** (attempt - 1))
                print(
                    f"OpenRouter multimodal request failed (attempt {attempt}/{retry_count}): {e},"
                    f"retrying in {sleep_s:.1f}s..."
                )
                time.sleep(sleep_s)
                continue
            break

    if last_error is not None:
        raise last_error
    return None


def _call_openrouter_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    reference_image: Optional[Image.Image] = None,
) -> Optional[Image.Image]:
    """Call OpenRouter image generation API using requests"""
    api_url = _get_openrouter_api_url(base_url)
    headers = _get_openrouter_headers(api_key)

    if reference_image is None:
        messages = [{'role': 'user', 'content': prompt}]
    else:
        buf = io.BytesIO()
        reference_image.save(buf, format='PNG')
        image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        message_content: List[Dict[str, Any]] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
        messages = [{'role': 'user', 'content': message_content}]

    payload = {
        'model': model,
        'messages': messages,
        # For OpenRouter Gemini image models, forcing image-only significantly reduces the chance of text-only responses
        'modalities': ['image'],
        'stream': False
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=300)

    if response.status_code != 200:
        raise Exception(f'OpenRouter API error: {response.status_code} - {response.text[:500]}')

    result = response.json()

    if 'error' in result:
        error_msg = result.get('error', {})
        if isinstance(error_msg, dict):
            error_msg = error_msg.get('message', str(error_msg))
        raise Exception(f'OpenRouter API error: {error_msg}')

    def _extract_data_url_payload(data_url: str) -> Optional[str]:
        match = re.match(r"^data:image/[^;]+;base64,(.+)$", data_url, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        return re.sub(r"\s+", "", match.group(1))

    def _decode_base64_image(image_b64: str) -> Optional[Image.Image]:
        if not image_b64:
            return None
        try:
            b64 = re.sub(r"\s+", "", image_b64)
            padding = len(b64) % 4
            if padding:
                b64 += "=" * (4 - padding)
            image_data = base64.b64decode(b64)
            image = Image.open(io.BytesIO(image_data))
            image.load()
            return image
        except Exception:
            return None

    def _load_remote_image(image_url: str) -> Optional[Image.Image]:
        try:
            resp = requests.get(image_url, timeout=120)
            if resp.status_code != 200 or not resp.content:
                return None
            image = Image.open(io.BytesIO(resp.content))
            image.load()
            return image
        except Exception:
            return None

    def _extract_image_url(value: Any) -> Optional[str]:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            if isinstance(value.get("url"), str):
                return value["url"]
            if "image_url" in value:
                return _extract_image_url(value.get("image_url"))
        return None

    def _try_parse_image_candidate(candidate: Any) -> Optional[Image.Image]:
        if isinstance(candidate, dict):
            # Common image fields for OpenAI/OpenRouter
            for key in ("b64_json", "base64", "data"):
                raw = candidate.get(key)
                if isinstance(raw, str):
                    parsed = _decode_base64_image(raw)
                    if parsed is not None:
                        return parsed
            if "image_url" in candidate:
                parsed = _try_parse_image_candidate(candidate.get("image_url"))
                if parsed is not None:
                    return parsed
            if "url" in candidate:
                parsed = _try_parse_image_candidate(candidate.get("url"))
                if parsed is not None:
                    return parsed
            return None

        if not isinstance(candidate, str):
            return None

        candidate = candidate.strip()
        if not candidate:
            return None

        if candidate.startswith("data:image/"):
            b64_payload = _extract_data_url_payload(candidate)
            if b64_payload:
                return _decode_base64_image(b64_payload)
            return None

        if candidate.startswith("http://") or candidate.startswith("https://"):
            return _load_remote_image(candidate)

        # In rare cases the service returns raw base64 directly
        return _decode_base64_image(candidate)

    def _extract_markdown_image_urls(text: str) -> list[str]:
        urls: list[str] = []
        for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", text):
            urls.append(match.group(1).strip())
        for match in re.finditer(r"data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+", text, flags=re.IGNORECASE):
            urls.append(match.group(0).strip())
        return urls

    choices = result.get('choices', [])
    if not choices:
        raise RuntimeError("OpenRouter response has no choices, cannot parse image result.")

    message = choices[0].get('message', {})
    candidates: list[Any] = []

    images = message.get("images")
    if isinstance(images, list):
        candidates.extend(images)
    elif images is not None:
        candidates.append(images)

    content = message.get("content")
    if isinstance(content, list):
        candidates.extend(content)
    elif isinstance(content, str):
        candidates.extend(_extract_markdown_image_urls(content))

    # Some proxy layers put the image in top-level fields
    top_images = result.get("images")
    if isinstance(top_images, list):
        candidates.extend(top_images)

    for item in candidates:
        # Try parsing the object directly first
        parsed = _try_parse_image_candidate(item)
        if parsed is not None:
            return parsed

        # Then try extracting a URL string from the object
        image_url = _extract_image_url(item)
        if image_url:
            parsed = _try_parse_image_candidate(image_url)
            if parsed is not None:
                return parsed

    content_preview = ""
    if isinstance(content, str):
        content_preview = content[:240].replace("\n", " ")

    refusal = message.get("refusal")
    message_keys = sorted(message.keys()) if isinstance(message, dict) else []
    images_count = len(images) if isinstance(images, list) else 0

    raise RuntimeError(
        "OpenRouter response succeeded but contained no parseable image."
        f" model={model}, message_keys={message_keys}, images_count={images_count}, "
        f"content_type={type(content).__name__}, refusal={refusal!r}, "
        f"content_preview={content_preview!r}"
    )


# ============================================================================
# Gemini provider implementation (Google official SDK)
# ============================================================================

def _get_gemini_client(api_key: str):
    """Get Gemini client (lazy import to avoid hard dependency in non-Gemini scenarios)"""
    try:
        from google import genai
    except ImportError as e:
        raise ImportError(
            "google-genai not installed, please run: pip install google-genai"
        ) from e
    return genai.Client(api_key=api_key)


def _build_gemini_text_config(max_tokens: int, temperature: float):
    """Build Gemini text generation config"""
    from google.genai import types

    return types.GenerateContentConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )


def _extract_gemini_text(response: Any) -> Optional[str]:
    """Extract text from Gemini response"""
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    parts = getattr(response, "parts", None) or []
    extracted: list[str] = []
    for part in parts:
        part_text = getattr(part, "text", None)
        if isinstance(part_text, str) and part_text.strip():
            extracted.append(part_text)
    if extracted:
        return "\n".join(extracted)

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        candidate_parts = getattr(content, "parts", None) or []
        for part in candidate_parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                extracted.append(part_text)
    if extracted:
        return "\n".join(extracted)

    return None


def _extract_gemini_image(response: Any) -> Optional[Image.Image]:
    """Extract image from Gemini response (prefer part.as_image())"""
    parts = getattr(response, "parts", None) or []
    for part in parts:
        as_image = getattr(part, "as_image", None)
        if callable(as_image):
            image = as_image()
            if image is not None:
                return image

        inline_data = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
        if inline_data is None:
            continue
        data = getattr(inline_data, "data", None)
        if isinstance(data, bytes) and data:
            return Image.open(io.BytesIO(data))
        if isinstance(data, str) and data:
            return Image.open(io.BytesIO(base64.b64decode(data)))

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        candidate_parts = getattr(content, "parts", None) or []
        for part in candidate_parts:
            as_image = getattr(part, "as_image", None)
            if callable(as_image):
                image = as_image()
                if image is not None:
                    return image
    return None


def _call_gemini_text(
    prompt: str,
    api_key: str,
    model: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """Call Gemini text API"""
    try:
        client = _get_gemini_client(api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=_build_gemini_text_config(max_tokens=max_tokens, temperature=temperature),
        )
        return _extract_gemini_text(response)
    except Exception as e:
        print(f"[Gemini] Text API call failed: {e}")
        raise


def _call_gemini_multimodal(
    contents: List[Any],
    api_key: str,
    model: str,
    max_tokens: int = 16000,
    temperature: float = 0.7,
) -> Optional[str]:
    """Call Gemini multimodal API"""
    try:
        client = _get_gemini_client(api_key)
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=_build_gemini_text_config(max_tokens=max_tokens, temperature=temperature),
        )
        return _extract_gemini_text(response)
    except Exception as e:
        print(f"[Gemini] Multimodal API call failed: {e}")
        raise


def _call_gemini_image_generation(
    prompt: str,
    api_key: str,
    model: str,
    reference_image: Optional[Image.Image] = None,
    image_size: str = GEMINI_DEFAULT_IMAGE_SIZE,
) -> Optional[Image.Image]:
    """Call Gemini image generation API, default image_size=4K"""
    try:
        from google.genai import types

        client = _get_gemini_client(api_key)
        config = types.GenerateContentConfig(
            image_config=types.ImageConfig(image_size=image_size),
        )

        if reference_image is None:
            contents: list[Any] = [prompt]
        else:
            # Reference image first, prompt after, following Gemini multimodal input convention
            contents = [reference_image, prompt]

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        return _extract_gemini_image(response)
    except Exception as e:
        print(f"[Gemini] Image generation API call failed: {e}")
        raise


# ============================================================================
# Step 1: Call LLM to generate image
# ============================================================================

def generate_figure_from_method(
    method_text: str,
    output_path: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    use_reference_image: Optional[bool] = None,
    reference_image_path: Optional[str] = None,
    image_size: str = GEMINI_DEFAULT_IMAGE_SIZE,
) -> str:
    """
    Generate academic-style image using LLM

    Args:
        method_text: Paper method text content
        output_path: output image path
        api_key: API key
        model: image generation model name
        base_url: API base URL
        provider: API provider
        use_reference_image: whether to use reference image (None uses global setting)
        reference_image_path: reference image path (None uses global setting)

    Returns:
        generated image path
    """
    print("=" * 60)
    print("Step 1: Generating academic-style image using LLM")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    if provider == "gemini":
        print(f"Resolution: {image_size}")

    if use_reference_image is None:
        use_reference_image = USE_REFERENCE_IMAGE
    if reference_image_path is None:
        reference_image_path = REFERENCE_IMAGE_PATH
    if reference_image_path:
        use_reference_image = True

    reference_image = None
    if use_reference_image:
        if not reference_image_path:
            raise ValueError("Reference image mode enabled but reference_image_path not provided")
        reference_image = Image.open(reference_image_path)
        print(f"Reference image: {reference_image_path}")

    if use_reference_image:
        prompt = f"""Generate a figure to visualize the method described below.

You should closely imitate the visual (artistic) style of the reference figure I provide, focusing only on aesthetic aspects, NOT on layout or structure.

Specifically, match:
- overall visual tone and mood
- illustration abstraction level
- line style
- color usage
- shading style
- icon and shape style
- arrow and connector aesthetics
- typography feel

The content structure, number of components, and layout may differ freely.
Only the visual style should be consistent.

The goal is that the figure looks like it was drawn by the same illustrator using the same visual design language as the reference figure.

Below is the method section of the paper:
\"\"\"
{method_text}
\"\"\""""
    else:
        prompt = f"""Generate a professional academic journal style figure for the paper below so as to visualize the method it proposes, below is the method section of this paper:

{method_text}

The figure should be engaging and using academic journal style with cute characters."""

    print(f"Sending request to: {base_url}")

    img = call_llm_image_generation(
        prompt=prompt,
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=provider,
        reference_image=reference_image,
        image_size=image_size,
    )

    if img is None:
        raise Exception('No image found in API response')

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to PNG for saving (Gemini image object save() may not accept format parameter)
    try:
        img.save(str(output_path), format='PNG')
    except TypeError:
        img.save(str(output_path))
        # Some SDK objects save with their default encoding (e.g. JPEG); force re-save as real PNG
        with Image.open(str(output_path)) as normalized:
            normalized.save(str(output_path), format='PNG')
    print(f"Image saved: {output_path}")
    return str(output_path)


# ============================================================================
# Step 2: SAM3 segmentation + Box merge + gray fill+black border+numbered labels
# ============================================================================

def get_label_font(box_width: int, box_height: int) -> ImageFont.FreeTypeFont:
    """
    Dynamically calculate appropriate font size based on box dimensions

    Args:
        box_width: rectangle width
        box_height: rectangle height

    Returns:
        PIL ImageFont object
    """
    # Font size is 1/4 of the shorter box side, min 12, max 48
    min_dim = min(box_width, box_height)
    font_size = max(12, min(48, min_dim // 4))

    # Try to load font
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "C:/Windows/Fonts/arial.ttf",  # Windows
    ]

    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, font_size)
        except (IOError, OSError):
            continue

    # Fall back to default font
    try:
        return ImageFont.load_default()
    except:
        return None


# ============================================================================
# Box merge helper functions
# ============================================================================

def calculate_overlap_ratio(box1: dict, box2: dict) -> float:
    """
    Calculate overlap ratio between two boxes

    Args:
        box1: first box, containing x1, y1, x2, y2
        box2: second box, containing x1, y1, x2, y2

    Returns:
        overlap ratio = intersection area / smaller box area
    """
    # Calculate intersection region
    x1 = max(box1["x1"], box2["x1"])
    y1 = max(box1["y1"], box2["y1"])
    x2 = min(box1["x2"], box2["x2"])
    y2 = min(box1["y2"], box2["y2"])

    # No intersection
    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # Calculate individual areas
    area1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
    area2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])

    if area1 == 0 or area2 == 0:
        return 0.0

    # Return intersection as fraction of smaller box
    return intersection / min(area1, area2)


def merge_two_boxes(box1: dict, box2: dict) -> dict:
    """
    Merge two boxes into their minimum bounding rectangle

    Args:
        box1: first box
        box2: second box

    Returns:
        merged box (minimum bounding rectangle)
    """
    merged = {
        "x1": min(box1["x1"], box2["x1"]),
        "y1": min(box1["y1"], box2["y1"]),
        "x2": max(box1["x2"], box2["x2"]),
        "y2": max(box1["y2"], box2["y2"]),
        "score": max(box1.get("score", 0), box2.get("score", 0)),  # Keep higher confidence
    }
    # Merge prompt fields (if present)
    prompt1 = box1.get("prompt", "")
    prompt2 = box2.get("prompt", "")
    if prompt1 and prompt2:
        if prompt1 == prompt2:
            merged["prompt"] = prompt1
        else:
            # Merge different prompts, keep the one with higher confidence
            if box1.get("score", 0) >= box2.get("score", 0):
                merged["prompt"] = prompt1
            else:
                merged["prompt"] = prompt2
    elif prompt1:
        merged["prompt"] = prompt1
    elif prompt2:
        merged["prompt"] = prompt2
    return merged


def merge_overlapping_boxes(boxes: list, overlap_threshold: float = 0.9) -> list:
    """
    Iteratively merge overlapping boxes

    Args:
        boxes: list of boxes, each containing x1, y1, x2, y2, score
        overlap_threshold: overlap threshold, boxes above this value are merged (default 0.9)

    Returns:
        merged box list, renumbered
    """
    if overlap_threshold <= 0 or len(boxes) <= 1:
        return boxes

    # Copy list to avoid modifying original data
    working_boxes = [box.copy() for box in boxes]

    merged = True
    iteration = 0
    while merged:
        merged = False
        iteration += 1
        n = len(working_boxes)

        for i in range(n):
            if merged:
                break
            for j in range(i + 1, n):
                ratio = calculate_overlap_ratio(working_boxes[i], working_boxes[j])
                if ratio >= overlap_threshold:
                    # Merge box_i and box_j
                    new_box = merge_two_boxes(working_boxes[i], working_boxes[j])
                    # Remove original two boxes, add merged box
                    working_boxes = [
                        working_boxes[k] for k in range(n) if k != i and k != j
                    ]
                    working_boxes.append(new_box)
                    merged = True
                    print(f"    Iteration {iteration}: merged box {i} and box {j} (overlap ratio: {ratio:.2f})")
                    break

    # Renumber
    result = []
    for idx, box in enumerate(working_boxes):
        result_box = {
            "id": idx,
            "label": f"<AF>{idx + 1:02d}",
            "x1": box["x1"],
            "y1": box["y1"],
            "x2": box["x2"],
            "y2": box["y2"],
            "score": box.get("score", 0),
        }
        # Keep prompt field (if present)
        if "prompt" in box:
            result_box["prompt"] = box["prompt"]
        result.append(result_box)

    return result


def _get_fal_api_key(sam_api_key: Optional[str]) -> str:
    key = sam_api_key or os.environ.get("FAL_KEY")
    if not key:
        raise ValueError("SAM3 fal.ai API key missing: set --sam_api_key or FAL_KEY environment variable")
    return key


def _get_roboflow_api_key(sam_api_key: Optional[str]) -> str:
    key = sam_api_key or os.environ.get("ROBOFLOW_API_KEY") or os.environ.get("API_KEY")
    if not key:
        raise ValueError(
            "SAM3 Roboflow API key missing: set --sam_api_key or ROBOFLOW_API_KEY/API_KEY environment variable"
        )
    return key


def _image_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_b64}"


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _cxcywh_norm_to_xyxy(box: list | tuple, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    if not box or len(box) < 4:
        return None
    try:
        cx, cy, bw, bh = [float(v) for v in box[:4]]
    except (TypeError, ValueError):
        return None

    cx *= width
    cy *= height
    bw *= width
    bh *= height

    x1 = int(round(cx - bw / 2.0))
    y1 = int(round(cy - bh / 2.0))
    x2 = int(round(cx + bw / 2.0))
    y2 = int(round(cy + bh / 2.0))

    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _polygon_to_bbox(points: list, width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    xs: list[float] = []
    ys: list[float] = []

    for pt in points:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        try:
            x = float(pt[0])
            y = float(pt[1])
        except (TypeError, ValueError):
            continue
        xs.append(x)
        ys.append(y)

    if not xs or not ys:
        return None

    x1 = int(round(min(xs)))
    y1 = int(round(min(ys)))
    x2 = int(round(max(xs)))
    y2 = int(round(max(ys)))

    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _extract_sam3_api_detections(response_json: dict, image_size: tuple[int, int]) -> list[dict]:
    width, height = image_size
    detections: list[dict] = []

    metadata = response_json.get("metadata") if isinstance(response_json, dict) else None
    if isinstance(metadata, list) and metadata:
        for item in metadata:
            if not isinstance(item, dict):
                continue
            box = item.get("box")
            xyxy = _cxcywh_norm_to_xyxy(box, width, height)
            if not xyxy:
                continue
            score = item.get("score")
            detections.append(
                {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3], "score": score}
            )
        return detections

    boxes = response_json.get("boxes") if isinstance(response_json, dict) else None
    scores = response_json.get("scores") if isinstance(response_json, dict) else None
    if isinstance(boxes, list) and boxes:
        scores_list = scores if isinstance(scores, list) else []
        for idx, box in enumerate(boxes):
            xyxy = _cxcywh_norm_to_xyxy(box, width, height)
            if not xyxy:
                continue
            score = scores_list[idx] if idx < len(scores_list) else None
            detections.append(
                {"x1": xyxy[0], "y1": xyxy[1], "x2": xyxy[2], "y2": xyxy[3], "score": score}
            )

    return detections


def _extract_roboflow_detections(response_json: dict, image_size: tuple[int, int]) -> list[dict]:
    width, height = image_size
    detections: list[dict] = []

    prompt_results = response_json.get("prompt_results") if isinstance(response_json, dict) else None
    if not isinstance(prompt_results, list):
        return detections

    for prompt_result in prompt_results:
        if not isinstance(prompt_result, dict):
            continue
        predictions = prompt_result.get("predictions", [])
        if not isinstance(predictions, list):
            continue
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            confidence = prediction.get("confidence")
            masks = prediction.get("masks", [])
            if not isinstance(masks, list):
                continue
            for mask in masks:
                points = []
                if isinstance(mask, list) and mask:
                    if isinstance(mask[0], (list, tuple)) and len(mask[0]) >= 2 and isinstance(
                        mask[0][0], (int, float)
                    ):
                        points = mask
                    elif isinstance(mask[0], (list, tuple)):
                        for sub in mask:
                            if isinstance(sub, (list, tuple)) and len(sub) >= 2 and isinstance(
                                sub[0], (int, float)
                            ):
                                points.append(sub)
                            elif isinstance(sub, (list, tuple)) and sub and isinstance(
                                sub[0], (list, tuple)
                            ):
                                for pt in sub:
                                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                                        points.append(pt)
                if not points:
                    continue
                xyxy = _polygon_to_bbox(points, width, height)
                if not xyxy:
                    continue
                detections.append(
                    {
                        "x1": xyxy[0],
                        "y1": xyxy[1],
                        "x2": xyxy[2],
                        "y2": xyxy[3],
                        "score": confidence,
                    }
                )

    return detections


def _call_sam3_api(
    image_data_uri: str,
    prompt: str,
    api_key: str,
    max_masks: int,
) -> dict:
    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "image_url": image_data_uri,
        "prompt": prompt,
        "apply_mask": False,
        "return_multiple_masks": True,
        "max_masks": max_masks,
        "include_scores": True,
        "include_boxes": True,
    }
    response = requests.post(SAM3_FAL_API_URL, headers=headers, json=payload, timeout=SAM3_API_TIMEOUT)
    if response.status_code != 200:
        raise Exception(f"SAM3 API error: {response.status_code} - {response.text[:500]}")
    result = response.json()
    if isinstance(result, dict) and "error" in result:
        raise Exception(f"SAM3 API error: {result.get('error')}")
    return result


def _call_sam3_roboflow_api(
    image_base64: str,
    prompt: str,
    api_key: str,
    min_score: float,
) -> dict:
    def _redact_secret(text: str) -> str:
        if not api_key:
            return text
        return text.replace(api_key, "***")

    payload = {
        "image": {"type": "base64", "value": image_base64},
        "prompts": [{"type": "text", "text": prompt}],
        "format": "polygon",
        "output_prob_thresh": min_score,
    }
    def _is_dns_error(exc: Exception) -> bool:
        msg = str(exc)
        patterns = [
            "NameResolutionError",
            "Temporary failure in name resolution",
            "getaddrinfo failed",
            "nodename nor servname provided",
            "gaierror",
        ]
        return any(p in msg for p in patterns)

    fallback_urls_env = os.environ.get("ROBOFLOW_API_FALLBACK_URLS", "")
    fallback_urls = [u.strip() for u in fallback_urls_env.split(",") if u.strip()]
    endpoint_urls = [SAM3_ROBOFLOW_API_URL] + [u for u in fallback_urls if u != SAM3_ROBOFLOW_API_URL]

    retry_count_env = os.environ.get("SAM3_API_RETRIES", "3")
    retry_delay_env = os.environ.get("SAM3_API_RETRY_DELAY", "1.5")
    try:
        retry_count = max(1, int(retry_count_env))
    except ValueError:
        retry_count = 3
    try:
        retry_delay = max(0.0, float(retry_delay_env))
    except ValueError:
        retry_delay = 1.5

    last_error: Optional[Exception] = None

    for endpoint in endpoint_urls:
        url = f"{endpoint}?api_key={api_key}"
        for attempt in range(1, retry_count + 1):
            try:
                response = requests.post(url, json=payload, timeout=SAM3_API_TIMEOUT)
                if response.status_code != 200:
                    raise Exception(
                        f"SAM3 Roboflow API error: {response.status_code} - {response.text[:500]}"
                    )
                result = response.json()
                if isinstance(result, dict) and "error" in result:
                    raise Exception(f"SAM3 Roboflow API error: {result.get('error')}")
                return result
            except requests.exceptions.RequestException as e:
                last_error = e
                # Exponential backoff retry for intermittent DNS/network issues
                if attempt < retry_count:
                    sleep_s = retry_delay * (2 ** (attempt - 1))
                    safe_error = _redact_secret(str(e))
                    print(
                        f"    Roboflow request failed (attempt {attempt}/{retry_count}): {safe_error},"
                        f"retrying in {sleep_s:.1f}s..."
                    )
                    time.sleep(sleep_s)
                    continue
                # Current endpoint retries exhausted, switching to next endpoint
                break
            except Exception as e:
                last_error = e
                break

    if last_error is not None and _is_dns_error(last_error):
        raise RuntimeError(
            "SAM3 Roboflow DNS resolution failed (container DNS cannot resolve serverless.roboflow.com).\n"
            "Available fixes:\n"
            "1) Set dns in docker-compose.yml (e.g. 223.5.5.5 / 119.29.29.29);\n"
            "2) Set ROBOFLOW_API_URL or ROBOFLOW_API_FALLBACK_URLS in .env;\n"
            "3) Temporarily switch to --sam_backend fal (requires FAL_KEY)."
        ) from last_error

    if last_error is not None:
        raise RuntimeError(f"SAM3 Roboflow request failed: {_redact_secret(str(last_error))}") from last_error

    raise RuntimeError("SAM3 Roboflow request failed: unknown error")


def segment_with_sam3(
    image_path: str,
    output_dir: str,
    text_prompts: str = "icon",
    min_score: float = 0.5,
    merge_threshold: float = 0.9,
    sam_backend: Literal["local", "fal", "roboflow", "api"] = "local",
    sam_api_key: Optional[str] = None,
    sam_max_masks: int = 32,
) -> tuple[str, str, list]:
    """
    Segment image using SAM3 with gray fill+black border+numbered labels, generate boxlib.json

    Placeholder style:
    - Gray fill (#808080)
    - Black border (width=3)
    - White centered numbered labels (<AF>01, <AF>02, ...)

    Args:
        image_path: input image path
        output_dir: output directory
        text_prompts: SAM3 text prompts, supports comma-separated multiple prompts (e.g. "icon,diagram,arrow")
        min_score: minimum confidence threshold
        merge_threshold: box merge threshold, boxes with overlap above this value are merged (0=no merge, default 0.9)

    Returns:
        (samed_path, boxlib_path, valid_boxes)
    """
    print("\n" + "=" * 60)
    print("Step 2: SAM3 segmentation + gray fill+black border+numbered labels")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path)
    original_size = image.size
    print(f"Original image size: {original_size[0]} x {original_size[1]}")

    # Parse multiple prompts (comma-separated)
    prompt_list = [p.strip() for p in text_prompts.split(",") if p.strip()]
    print(f"Using prompts: {prompt_list}")

    # Detect separately for each prompt and collect results
    all_detected_boxes = []
    total_detected = 0

    backend = sam_backend
    if backend == "api":
        backend = "fal"

    if backend == "local":
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        import sam3

        sam3_dir = Path(sam3.__path__[0]) if hasattr(sam3, '__path__') else Path(sam3.__file__).parent
        bpe_path = sam3_dir / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        if not bpe_path.exists():
            bpe_path = None
            print("Warning: bpe file not found, using default path")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model = build_sam3_image_model(device=device, bpe_path=str(bpe_path) if bpe_path else None)
        processor = Sam3Processor(model, device=device)
        inference_state = processor.set_image(image)

        for prompt in prompt_list:
            print(f"\n  Detecting: '{prompt}'")
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)

            boxes = output["boxes"]
            scores = output["scores"]

            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()

            prompt_count = 0
            for box, score in zip(boxes, scores):
                if score >= min_score:
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    all_detected_boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "score": float(score),
                        "prompt": prompt  # record source prompt
                    })
                    prompt_count += 1
                    print(f"    Object {prompt_count}: ({x1}, {y1}, {x2}, {y2}), score={score:.3f}")
                else:
                    print(f"    Skipped: score={score:.3f} < {min_score}")

            print(f"  '{prompt}' detected {prompt_count} valid objects")
            total_detected += prompt_count

        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elif backend == "fal":
        api_key = _get_fal_api_key(sam_api_key)
        max_masks = max(1, min(32, int(sam_max_masks)))
        image_data_uri = _image_to_data_uri(image)
        print(f"SAM3 fal.ai API mode: max_masks={max_masks}")

        for prompt in prompt_list:
            print(f"\n  Detecting: '{prompt}'")
            response_json = _call_sam3_api(
                image_data_uri=image_data_uri,
                prompt=prompt,
                api_key=api_key,
                max_masks=max_masks,
            )
            detections = _extract_sam3_api_detections(response_json, original_size)
            prompt_count = 0
            for det in detections:
                score = det.get("score")
                score_val = float(score) if score is not None else 0.0
                if score_val >= min_score:
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    all_detected_boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "score": score_val,
                        "prompt": prompt  # record source prompt
                    })
                    prompt_count += 1
                    print(f"    Object {prompt_count}: ({x1}, {y1}, {x2}, {y2}), score={score_val:.3f}")
                else:
                    print(f"    Skipped: score={score_val:.3f} < {min_score}")

            print(f"  '{prompt}' detected {prompt_count} valid objects")
            total_detected += prompt_count
    elif backend == "roboflow":
        api_key = _get_roboflow_api_key(sam_api_key)
        image_base64 = _image_to_base64(image)
        print("SAM3 Roboflow API mode: format=polygon")

        for prompt in prompt_list:
            print(f"\n  Detecting: '{prompt}'")
            response_json = _call_sam3_roboflow_api(
                image_base64=image_base64,
                prompt=prompt,
                api_key=api_key,
                min_score=min_score,
            )
            detections = _extract_roboflow_detections(response_json, original_size)
            prompt_count = 0
            for det in detections:
                score = det.get("score")
                score_val = float(score) if score is not None else 0.0
                if score_val >= min_score:
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    all_detected_boxes.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "score": score_val,
                        "prompt": prompt
                    })
                    prompt_count += 1
                    print(f"    Object {prompt_count}: ({x1}, {y1}, {x2}, {y2}), score={score_val:.3f}")
                else:
                    print(f"    Skipped: score={score_val:.3f} < {min_score}")

            print(f"  '{prompt}' detected {prompt_count} valid objects")
            total_detected += prompt_count
    else:
        raise ValueError(f"Unknown SAM3 backend: {sam_backend}")

    print(f"\nTotal detected: {total_detected} objects (from {len(prompt_list)} prompts)")

    # Assign temporary id and label to all detected boxes (for merging)
    valid_boxes = []
    for i, box_data in enumerate(all_detected_boxes):
        valid_boxes.append({
            "id": i,
            "label": f"<AF>{i + 1:02d}",
            "x1": box_data["x1"],
            "y1": box_data["y1"],
            "x2": box_data["x2"],
            "y2": box_data["y2"],
            "score": box_data["score"],
            "prompt": box_data["prompt"]
        })

    # === Box merge step ===
    if merge_threshold > 0 and len(valid_boxes) > 1:
        print(f"\n  Merging overlapping boxes (threshold: {merge_threshold})...")
        original_count = len(valid_boxes)
        valid_boxes = merge_overlapping_boxes(valid_boxes, merge_threshold)
        merged_count = original_count - len(valid_boxes)
        if merged_count > 0:
            print(f"  Merge complete: {original_count} -> {len(valid_boxes)} ({merged_count} merged)")
            # Print merged box info
            print(f"\n  Merged boxes:")
            for box_info in valid_boxes:
                print(f"    {box_info['label']}: ({box_info['x1']}, {box_info['y1']}, {box_info['x2']}, {box_info['y2']})")
        else:
            print(f"  No merge needed, all box overlap ratios below threshold")

    # Create labeled image using merged valid_boxes
    print(f"\n  Drawing samed.png (using {len(valid_boxes)} boxes)...")
    samed_image = image.copy()
    draw = ImageDraw.Draw(samed_image)

    for box_info in valid_boxes:
        x1, y1, x2, y2 = box_info["x1"], box_info["y1"], box_info["x2"], box_info["y2"]
        label = box_info["label"]

        # Gray fill + black border
        draw.rectangle([x1, y1, x2, y2], fill="#808080", outline="black", width=3)

        # Calculate center point
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Get appropriately sized font
        box_width = x2 - x1
        box_height = y2 - y1
        font = get_label_font(box_width, box_height)

        # Draw white centered numbered label
        if font:
            # Use anchor="mm" for centered drawing (if supported)
            try:
                draw.text((cx, cy), label, fill="white", anchor="mm", font=font)
            except TypeError:
                # Older PIL doesn't support anchor, calculate position manually
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = cx - text_width // 2
                text_y = cy - text_height // 2
                draw.text((text_x, text_y), label, fill="white", font=font)
        else:
            # Use default when no font available
            draw.text((cx, cy), label, fill="white")

    samed_path = output_dir / "samed.png"
    samed_image.save(str(samed_path))
    print(f"Labeled image saved: {samed_path}")

    boxlib_data = {
        "image_size": {"width": original_size[0], "height": original_size[1]},
        "prompts_used": prompt_list,
        "boxes": valid_boxes,
        BOXLIB_NO_ICON_MODE_KEY: len(valid_boxes) == 0,
    }

    boxlib_path = output_dir / "boxlib.json"
    with open(boxlib_path, 'w', encoding='utf-8') as f:
        json.dump(boxlib_data, f, indent=2, ensure_ascii=False)
    print(f"Box info saved: {boxlib_path}")

    return str(samed_path), str(boxlib_path), valid_boxes


# ============================================================================
# Step 3: Crop + RMBG2 background removal
# ============================================================================

def _get_hf_token() -> Optional[str]:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    if not isinstance(token, str):
        return None
    token = token.strip()
    return token or None


def _has_rmbg2_cached_weights() -> bool:
    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    snapshots_dir = hf_home / "hub" / "models--briaai--RMBG-2.0" / "snapshots"
    if not snapshots_dir.exists():
        return False
    return any(snapshots_dir.glob("*/config.json"))


def _ensure_rmbg2_access_ready(rmbg_model_path: Optional[str]) -> None:
    if rmbg_model_path and Path(rmbg_model_path).exists():
        return
    if _get_hf_token() is not None:
        return
    if _has_rmbg2_cached_weights():
        return
    raise RuntimeError(
        "Step 3 requires briaai/RMBG-2.0, but no valid access credentials were found.\n"
        "Please complete the following:\n"
        "1) Request access at https://huggingface.co/briaai/RMBG-2.0\n"
        "2) Set HF_TOKEN=your_read_token in .env\n"
        "3) Re-run: docker compose up -d --build"
    )


class BriaRMBG2Remover:
    """High-quality background removal using BRIA-RMBG 2.0 model"""

    def __init__(self, model_path: Path | str | None = None, output_dir: Path | str | None = None):
        self.model_path = Path(model_path) if model_path else None
        self.output_dir = Path(output_dir) if output_dir else Path("./output/icons")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_repo_id = "briaai/RMBG-2.0"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        hf_token = _get_hf_token()

        if self.model_path and self.model_path.exists():
            print(f"Loading local RMBG weights: {self.model_path}")
            self.model = AutoModelForImageSegmentation.from_pretrained(
                str(self.model_path), trust_remote_code=True,
            ).eval().to(device)
        else:
            print("Loading RMBG-2.0 model from HuggingFace...")
            if hf_token:
                print("HF_TOKEN detected, using authenticated access for gated model.")
            else:
                print("HF_TOKEN not detected, attempting anonymous access (will usually fail for gated model).")

            try:
                self.model = AutoModelForImageSegmentation.from_pretrained(
                    self.model_repo_id,
                    trust_remote_code=True,
                    token=hf_token,
                ).eval().to(device)
            except Exception as e:
                msg = str(e).lower()
                is_gated = (
                    "gated repo" in msg
                    or "cannot access gated repo" in msg
                    or "access to model briaai/rmbg-2.0 is restricted" in msg
                    or "401 client error" in msg
                    or "you are trying to access a gated repo" in msg
                )
                if is_gated:
                    raise RuntimeError(
                        "Cannot download RMBG-2.0 (HuggingFace gated model authentication failed).\n"
                        "Please configure as follows:\n"
                        "1) Log in and request model access: https://huggingface.co/briaai/RMBG-2.0\n"
                        "2) Create a token with Read permissions\n"
                        "3) Set HF_TOKEN=your_token in the project .env\n"
                        "4) Re-run: docker compose up -d --build"
                    ) from e
                raise

        self.image_size = (1024, 1024)
        self.transform_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def remove_background(self, image: Image.Image, output_name: str) -> str:
        image_rgb = image.convert("RGB")
        input_tensor = self.transform_image(image_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            preds = self.model(input_tensor)[-1].sigmoid().cpu()

        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_rgb.size)

        out = image_rgb.copy()
        out.putalpha(mask)

        out_path = self.output_dir / f"{output_name}_nobg.png"
        out.save(out_path)
        return str(out_path)


def crop_and_remove_background(
    image_path: str,
    boxlib_path: str,
    output_dir: str,
    rmbg_model_path: Optional[str] = None,
) -> list[dict]:
    """
    Crop image according to boxlib.json and remove background using RMBG2

    Files named by label: icon_AF01.png, icon_AF01_nobg.png
    """
    print("\n" + "=" * 60)
    print("Step 3: Crop + RMBG2 background removal")
    print("=" * 60)

    output_dir = Path(output_dir)
    icons_dir = output_dir / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path)
    with open(boxlib_path, 'r', encoding='utf-8') as f:
        boxlib_data = json.load(f)

    boxes = boxlib_data["boxes"]

    if len(boxes) == 0:
        print("Warning: no valid boxes detected")
        return []

    remover = BriaRMBG2Remover(model_path=rmbg_model_path, output_dir=icons_dir)

    icon_infos = []
    for box_info in boxes:
        box_id = box_info["id"]
        label = box_info.get("label", f"<AF>{box_id + 1:02d}")
        # Convert <AF>01 to AF01 for filename
        label_clean = label.replace("<", "").replace(">", "")

        x1, y1, x2, y2 = box_info["x1"], box_info["y1"], box_info["x2"], box_info["y2"]

        cropped = image.crop((x1, y1, x2, y2))
        crop_path = icons_dir / f"icon_{label_clean}.png"
        cropped.save(crop_path)

        nobg_path = remover.remove_background(cropped, f"icon_{label_clean}")

        icon_infos.append({
            "id": box_id,
            "label": label,
            "label_clean": label_clean,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": x2 - x1, "height": y2 - y1,
            "crop_path": str(crop_path),
            "nobg_path": nobg_path,
        })

        print(f"  {label}: crop and background removal done -> {nobg_path}")

    del remover
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return icon_infos


# ============================================================================
# Step 4: Multimodal call to generate SVG
# ============================================================================

def generate_svg_template(
    figure_path: str,
    samed_path: str,
    boxlib_path: str,
    output_path: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    placeholder_mode: PlaceholderMode = "label",
    no_icon_mode: bool = False,
) -> str:
    """
    Generate SVG code using multimodal LLM

    Args:
        placeholder_mode: placeholder mode
            - "none": no special style
            - "box": pass boxlib coordinates
            - "label": gray fill+black border+numbered labels (recommended)
    """
    print("\n" + "=" * 60)
    print("Step 4: Multimodal call to generate SVG")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Placeholder mode: {placeholder_mode}")
    if no_icon_mode:
        print("No-icon mode: enabling pure SVG reproduction fallback")

    figure_img = Image.open(figure_path)
    samed_img = Image.open(samed_path)

    figure_width, figure_height = figure_img.size
    print(f"Original image size: {figure_width} x {figure_height}")

    if no_icon_mode:
        prompt_text = f"""Write SVG code to reproduce this image as closely as possible at the pixel level.

SAM3 did not detect any valid icons, so this is a no-icon fallback mode task:
- Do not add any gray rectangle placeholders
- Do not add any <AF>01 / <AF>02 labels
- Do not generate phantom icon boxes, placeholder groups or extra decorations
- All visible content should be reproduced directly using SVG elements
- Prioritize matching the overall layout, text, arrows, lines, borders, and colors of the original image

CRITICAL DIMENSION REQUIREMENT:
- The original image has dimensions: {figure_width} x {figure_height} pixels
- Your SVG MUST use these EXACT dimensions:
  - Set viewBox="0 0 {figure_width} {figure_height}"
  - Set width="{figure_width}" height="{figure_height}"
- DO NOT scale or resize the SVG

Image reference notes:
- Image 1 is the original target figure.
- Image 2 is the SAM reference image. It does not contain any valid icon placeholder boxes for this run.

Please output ONLY the SVG code, starting with <svg and ending with </svg>. Do not include any explanation or markdown formatting."""
    else:
        # Base prompt
        base_prompt = f"""Write SVG code to reproduce this image at pixel level (all text and components (especially arrow styles) must match, except icons which should be replaced by same-sized rectangle placeholders (i.e. the content covered by gray rectangles is the icon area))

CRITICAL DIMENSION REQUIREMENT:
- The original image has dimensions: {figure_width} x {figure_height} pixels
- Your SVG MUST use these EXACT dimensions to ensure accurate icon placement:
  - Set viewBox="0 0 {figure_width} {figure_height}"
  - Set width="{figure_width}" height="{figure_height}"
- DO NOT scale or resize the SVG
"""

    if not no_icon_mode and placeholder_mode == "box":
        # box mode: pass boxlib coordinates
        with open(boxlib_path, 'r', encoding='utf-8') as f:
            boxlib_content = f.read()

        prompt_text = base_prompt + f"""
ICON COORDINATES FROM boxlib.json:
The following JSON contains precise icon coordinates detected by SAM3:
{boxlib_content}
Use these coordinates to accurately position your icon placeholders in the SVG.

Please output ONLY the SVG code, starting with <svg and ending with </svg>. Do not include any explanation or markdown formatting."""

    elif not no_icon_mode and placeholder_mode == "label":
        # label mode: require placeholder style to match samed.png
        prompt_text = base_prompt + """
PLACEHOLDER STYLE REQUIREMENT:
Look at the second image (samed.png) - each icon area is marked with a gray rectangle (#808080), black border, and a centered label like <AF>01, <AF>02, etc.

Your SVG placeholders MUST match this exact style:
- Rectangle with fill="#808080" and stroke="black" stroke-width="2"
- Centered white text showing the same label (<AF>01, <AF>02, etc.)
- Wrap each placeholder in a <g> element with id matching the label (e.g., id="AF01")

Example placeholder structure:
<g id="AF01">
  <rect x="100" y="50" width="80" height="80" fill="#808080" stroke="black" stroke-width="2"/>
  <text x="140" y="90" text-anchor="middle" dominant-baseline="middle" fill="white" font-size="14">&lt;AF&gt;01</text>
</g>

Please output ONLY the SVG code, starting with <svg and ending with </svg>. Do not include any explanation or markdown formatting."""

    elif not no_icon_mode:  # none mode
        prompt_text = base_prompt + """
Please output ONLY the SVG code, starting with <svg and ending with </svg>. Do not include any explanation or markdown formatting."""

    contents = [prompt_text, figure_img, samed_img]

    print(f"Sending multimodal request to: {base_url}")

    content = call_llm_multimodal(
        contents=contents,
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=provider,
        max_tokens=50000,
    )

    if not content:
        raise Exception(
            f"No content in API response (provider={provider}, model={model}).",
            "If using OpenRouter, try increasing OPENROUTER_MULTIMODAL_RETRIES and retrying.",
        )

    svg_code = extract_svg_code(content)

    if not svg_code:
        raise Exception('Could not extract SVG code from response')

    # Step 4.5: SVG syntax validation and repair
    svg_code = check_and_fix_svg(
        svg_code=svg_code,
        api_key=api_key,
        model=model,
        base_url=base_url,
        provider=provider,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_code)

    print(f"SVG template saved: {output_path}")
    return str(output_path)


def extract_svg_code(content: str) -> Optional[str]:
    """Extract SVG code from response content"""
    pattern = r'(<svg[\s\S]*?</svg>)'
    match = re.search(pattern, content, re.IGNORECASE)
    if match:
        return match.group(1)

    pattern = r'```(?:svg|xml)?\s*([\s\S]*?)```'
    match = re.search(pattern, content)
    if match:
        code = match.group(1).strip()
        if code.startswith('<svg'):
            return code

    if content.strip().startswith('<svg'):
        return content.strip()

    return None


# ============================================================================
# Step 4.5: SVG syntax validation and repair
# ============================================================================

def validate_svg_syntax(svg_code: str) -> tuple[bool, list[str]]:
    """Validate SVG syntax using lxml"""
    try:
        from lxml import etree
        etree.fromstring(svg_code.encode('utf-8'))
        return True, []
    except ImportError:
        print("  Warning: lxml not installed, using built-in xml.etree for validation")
        try:
            import xml.etree.ElementTree as ET
            ET.fromstring(svg_code)
            return True, []
        except ET.ParseError as e:
            return False, [f"XML parse error: {str(e)}"]
    except Exception as e:
        from lxml import etree
        if isinstance(e, etree.XMLSyntaxError):
            errors = []
            error_log = e.error_log
            for error in error_log:
                errors.append(f"Line {error.line}, col {error.column}: {error.message}")
            if not errors:
                errors.append(f"Line {e.lineno}, col {e.offset}: {e.msg}")
            return False, errors
        else:
            return False, [f"Parse error: {str(e)}"]


def fix_svg_with_llm(
    svg_code: str,
    errors: list[str],
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_retries: int = 3,
) -> str:
    """Fix SVG syntax errors using LLM"""
    print("\n  " + "-" * 50)
    print("  SVG syntax errors detected, calling LLM to repair...")
    print("  " + "-" * 50)
    for err in errors:
        print(f"    {err}")

    current_svg = svg_code
    current_errors = errors

    for attempt in range(max_retries):
        print(f"\n  Repair attempt {attempt + 1}/{max_retries}...")

        error_list = "\n".join([f"  - {err}" for err in current_errors])
        prompt = f"""The following SVG code has XML syntax errors detected by an XML parser. Please fix ALL the errors and return valid SVG code.

SYNTAX ERRORS DETECTED:
{error_list}

ORIGINAL SVG CODE:
```xml
{current_svg}
```

IMPORTANT INSTRUCTIONS:
1. Fix all XML syntax errors (unclosed tags, invalid attributes, unescaped characters, etc.)
2. Ensure the output is valid XML that can be parsed by lxml
3. Keep all the visual elements and structure intact
4. Return ONLY the fixed SVG code, starting with <svg and ending with </svg>
5. Do NOT include any markdown formatting, explanation, or code blocks - just the raw SVG code"""

        try:
            content = call_llm_text(
                prompt=prompt,
                api_key=api_key,
                model=model,
                base_url=base_url,
                provider=provider,
                max_tokens=16000,
                temperature=0.3,
            )

            if not content:
                print("    Response is empty")
                continue

            fixed_svg = extract_svg_code(content)

            if not fixed_svg:
                print("    Could not extract SVG code from response")
                continue

            is_valid, new_errors = validate_svg_syntax(fixed_svg)

            if is_valid:
                print("    Repair successful! SVG syntax validated")
                return fixed_svg
            else:
                print(f"    Still {len(new_errors)} errors after repair:")
                for err in new_errors[:3]:
                    print(f"      {err}")
                if len(new_errors) > 3:
                    print(f"      ... and {len(new_errors) - 3} more errors")
                current_svg = fixed_svg
                current_errors = new_errors

        except Exception as e:
            print(f"    Error during repair: {e}")
            continue

    print(f"  Warning: max retries reached ({max_retries}), returning last SVG code")
    return current_svg


def check_and_fix_svg(
    svg_code: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
) -> str:
    """Check SVG syntax and call LLM to repair if needed"""
    print("\n" + "-" * 50)
    print("Step 4.5: SVG syntax validation (using lxml XML parser)")
    print("-" * 50)

    is_valid, errors = validate_svg_syntax(svg_code)

    if is_valid:
        print("  SVG syntax validation passed!")
        return svg_code
    else:
        print(f"  Found {len(errors)} syntax errors")
        fixed_svg = fix_svg_with_llm(
            svg_code=svg_code,
            errors=errors,
            api_key=api_key,
            model=model,
            base_url=base_url,
            provider=provider,
        )
        return fixed_svg


# ============================================================================
# Step 4.7: Coordinate system alignment
# ============================================================================

def get_svg_dimensions(svg_code: str) -> tuple[Optional[float], Optional[float]]:
    """Extract coordinate system dimensions from SVG code"""
    viewbox_pattern = r'viewBox=["\']([^"\']+)["\']'
    viewbox_match = re.search(viewbox_pattern, svg_code, re.IGNORECASE)

    if viewbox_match:
        viewbox_value = viewbox_match.group(1).strip()
        parts = viewbox_value.split()
        if len(parts) >= 4:
            try:
                vb_width = float(parts[2])
                vb_height = float(parts[3])
                return vb_width, vb_height
            except ValueError:
                pass

    def parse_dimension(attr_name: str) -> Optional[float]:
        pattern = rf'{attr_name}=["\']([^"\']+)["\']'
        match = re.search(pattern, svg_code, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            numeric_match = re.match(r'([\d.]+)', value)
            if numeric_match:
                try:
                    return float(numeric_match.group(1))
                except ValueError:
                    pass
        return None

    width = parse_dimension('width')
    height = parse_dimension('height')

    if width and height:
        return width, height

    return None, None


def calculate_scale_factors(
    figure_width: int,
    figure_height: int,
    svg_width: float,
    svg_height: float,
) -> tuple[float, float]:
    """Calculate scale factors from figure.png pixel coordinates to SVG coordinates"""
    scale_x = svg_width / figure_width
    scale_y = svg_height / figure_height
    return scale_x, scale_y


# ============================================================================
# Step 5: Replace icons into SVG (supports label matching)
# ============================================================================

def replace_icons_in_svg(
    template_svg_path: str,
    icon_infos: list[dict],
    output_path: str,
    scale_factors: tuple[float, float] = (1.0, 1.0),
    match_by_label: bool = True,
) -> str:
    """
    Replace transparent-background icons into SVG placeholders

    Args:
        template_svg_path: SVG template path
        icon_infos: icon info list
        output_path: output path
        scale_factors: coordinate scale factors
        match_by_label: whether to use label matching (label mode)
    """
    print("\n" + "=" * 60)
    print("Step 5: Replacing icons into SVG")
    print("=" * 60)
    print(f"Match mode: {'label matching' if match_by_label else 'coordinate matching'}")

    scale_x, scale_y = scale_factors
    if scale_x != 1.0 or scale_y != 1.0:
        print(f"Applying coordinate scaling: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")

    with open(template_svg_path, 'r', encoding='utf-8') as f:
        svg_content = f.read()

    for icon_info in icon_infos:
        label = icon_info.get("label", "")
        label_clean = icon_info.get("label_clean", label.replace("<", "").replace(">", ""))
        nobg_path = icon_info["nobg_path"]

        # Read icon and convert to base64
        icon_img = Image.open(nobg_path)
        buf = io.BytesIO()
        icon_img.save(buf, format="PNG")
        icon_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        replaced = False

        if match_by_label and label:
            # Method 1: find <g> element with id="AF01"
            g_pattern = rf'<g[^>]*\bid=["\']?{re.escape(label_clean)}["\']?[^>]*>[\s\S]*?</g>'
            g_match = re.search(g_pattern, svg_content, re.IGNORECASE)

            if g_match:
                g_content = g_match.group(0)

                # Extract transform="translate(x, y)" from <g> element (if present)
                # This handles the case where LLM generates <g id="AF01" transform="translate(100, 50)"><rect x="0" y="0" ...>
                g_tag_match = re.match(r'<g[^>]*>', g_content, re.IGNORECASE)
                translate_x, translate_y = 0.0, 0.0
                if g_tag_match:
                    g_tag = g_tag_match.group(0)
                    # Match transform="translate(100, 50)" or transform="translate(100 50)"
                    transform_pattern = r'transform=["\'][^"\']*translate\s*\(\s*([\d.-]+)[\s,]+([\d.-]+)\s*\)'
                    transform_match = re.search(transform_pattern, g_tag, re.IGNORECASE)
                    if transform_match:
                        translate_x = float(transform_match.group(1))
                        translate_y = float(transform_match.group(2))

                # Extract <rect> dimensions from <g>
                rect_patterns = [
                    # x="100" y="50" width="80" height="80"
                    r'<rect[^>]*\bx=["\']?([\d.]+)["\']?[^>]*\by=["\']?([\d.]+)["\']?[^>]*\bwidth=["\']?([\d.]+)["\']?[^>]*\bheight=["\']?([\d.]+)["\']?',
                    # width="80" height="80" x="100" y="50" (attribute order may vary)
                    r'<rect[^>]*\bwidth=["\']?([\d.]+)["\']?[^>]*\bheight=["\']?([\d.]+)["\']?[^>]*\bx=["\']?([\d.]+)["\']?[^>]*\by=["\']?([\d.]+)["\']?',
                ]

                rect_info = None
                for rp in rect_patterns:
                    rect_match = re.search(rp, g_content, re.IGNORECASE)
                    if rect_match:
                        groups = rect_match.groups()
                        if len(groups) == 4:
                            if 'width' in rp[:50]:  # second pattern
                                width, height, x, y = groups
                            else:
                                x, y, width, height = groups
                            rect_info = {
                                'x': float(x),
                                'y': float(y),
                                'width': float(width),
                                'height': float(height)
                            }
                            break

                if rect_info:
                    # Add <g> transform translate values to rect coordinates
                    x = rect_info['x'] + translate_x
                    y = rect_info['y'] + translate_y
                    width, height = rect_info['width'], rect_info['height']

                    # Print notice if transform was applied
                    if translate_x != 0 or translate_y != 0:
                        print(f"  {label}: detected <g> transform: translate({translate_x}, {translate_y})")

                    # Create image tag to replace entire <g>
                    image_tag = f'<image id="icon_{label_clean}" x="{x}" y="{y}" width="{width}" height="{height}" href="data:image/png;base64,{icon_b64}" preserveAspectRatio="xMidYMid meet"/>'
                    svg_content = svg_content.replace(g_content, image_tag)
                    print(f"  {label}: replaced successfully (label match <g>) at ({x}, {y}) size {width}x{height}")
                    replaced = True

            # Method 2: find <rect> near <text> element containing label text
            if not replaced:
                # Find text containing <AF>01 or &lt;AF&gt;01
                text_patterns = [
                    rf'<text[^>]*>[^<]*{re.escape(label)}[^<]*</text>',
                    rf'<text[^>]*>[^<]*&lt;AF&gt;{label_clean[2:]}[^<]*</text>',
                ]

                for tp in text_patterns:
                    text_match = re.search(tp, svg_content, re.IGNORECASE)
                    if text_match:
                        # Found text, search backward for nearest <rect>
                        text_pos = text_match.start()
                        preceding_svg = svg_content[:text_pos]

                        # Find the last <rect>
                        rect_matches = list(re.finditer(r'<rect[^>]*/?\s*>', preceding_svg, re.IGNORECASE))
                        if rect_matches:
                            last_rect = rect_matches[-1]
                            rect_content = last_rect.group(0)

                            # Extract rect attributes
                            x_match = re.search(r'\bx=["\']?([\d.]+)', rect_content)
                            y_match = re.search(r'\by=["\']?([\d.]+)', rect_content)
                            w_match = re.search(r'\bwidth=["\']?([\d.]+)', rect_content)
                            h_match = re.search(r'\bheight=["\']?([\d.]+)', rect_content)

                            if all([x_match, y_match, w_match, h_match]):
                                x = float(x_match.group(1))
                                y = float(y_match.group(1))
                                width = float(w_match.group(1))
                                height = float(h_match.group(1))

                                # Replace rect and text
                                image_tag = f'<image id="icon_{label_clean}" x="{x}" y="{y}" width="{width}" height="{height}" href="data:image/png;base64,{icon_b64}" preserveAspectRatio="xMidYMid meet"/>'

                                # Delete text
                                svg_content = svg_content.replace(text_match.group(0), '')
                                # Replace rect
                                svg_content = svg_content.replace(rect_content, image_tag, 1)

                                print(f"  {label}: replaced successfully (label match <text>) at ({x}, {y}) size {width}x{height}")
                                replaced = True
                                break

        # Fallback: use coordinate matching
        if not replaced:
            orig_x1, orig_y1 = icon_info["x1"], icon_info["y1"]
            orig_width, orig_height = icon_info["width"], icon_info["height"]

            x1 = orig_x1 * scale_x
            y1 = orig_y1 * scale_y
            width = orig_width * scale_x
            height = orig_height * scale_y

            image_tag = f'<image id="icon_{label_clean}" x="{x1:.1f}" y="{y1:.1f}" width="{width:.1f}" height="{height:.1f}" href="data:image/png;base64,{icon_b64}" preserveAspectRatio="xMidYMid meet"/>'

            x1_int, y1_int = int(round(x1)), int(round(y1))

            # Exact match
            rect_pattern = rf'<rect[^>]*x=["\']?{x1_int}(?:\.0)?["\']?[^>]*y=["\']?{y1_int}(?:\.0)?["\']?[^>]*/?\s*>'
            if re.search(rect_pattern, svg_content):
                svg_content = re.sub(rect_pattern, image_tag, svg_content, count=1)
                print(f"  {label}: replaced successfully (exact coordinate match) at ({x1:.1f}, {y1:.1f})")
                replaced = True
            else:
                # Approximate match
                tolerance = 10
                found = False
                for dx in range(-tolerance, tolerance+1, 2):
                    for dy in range(-tolerance, tolerance+1, 2):
                        search_x = x1_int + dx
                        search_y = y1_int + dy
                        rect_pattern = rf'<rect[^>]*x=["\']?{search_x}(?:\.0)?["\']?[^>]*y=["\']?{search_y}(?:\.0)?["\']?[^>]*(?:fill=["\']?(?:#[0-9A-Fa-f]{{3,6}}|gray|grey)["\']?|stroke=["\']?(?:black|#000|#000000)["\']?)[^>]*/?\s*>'
                        if re.search(rect_pattern, svg_content, re.IGNORECASE):
                            svg_content = re.sub(rect_pattern, image_tag, svg_content, count=1, flags=re.IGNORECASE)
                            print(f"  {label}: replaced successfully (approximate coordinate match) at ({x1:.1f}, {y1:.1f})")
                            found = True
                            replaced = True
                            break
                    if found:
                        break

        if not replaced:
            # Append to end of SVG
            orig_x1, orig_y1 = icon_info["x1"], icon_info["y1"]
            orig_width, orig_height = icon_info["width"], icon_info["height"]
            x1 = orig_x1 * scale_x
            y1 = orig_y1 * scale_y
            width = orig_width * scale_x
            height = orig_height * scale_y

            image_tag = f'<image id="icon_{label_clean}" x="{x1:.1f}" y="{y1:.1f}" width="{width:.1f}" height="{height:.1f}" href="data:image/png;base64,{icon_b64}" preserveAspectRatio="xMidYMid meet"/>'
            svg_content = svg_content.replace('</svg>', f'  {image_tag}\n</svg>')
            print(f"  {label}: appended to SVG at ({x1:.1f}, {y1:.1f}) (no matching placeholder found)")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)

    print(f"Final SVG saved: {output_path}")
    return str(output_path)


# ============================================================================
# Step 4.6: LLM SVG optimization
# ============================================================================

def count_base64_images(svg_code: str) -> int:
    """Count embedded base64 images in SVG"""
    pattern = r'(?:href|xlink:href)=["\']data:image/[^;]+;base64,[A-Za-z0-9+/=]+'
    matches = re.findall(pattern, svg_code)
    return len(matches)


def validate_base64_images(svg_code: str, expected_count: int) -> tuple[bool, str]:
    """Verify that base64 images in SVG are complete"""
    actual_count = count_base64_images(svg_code)

    if actual_count < expected_count:
        return False, f"Insufficient base64 images: expected {expected_count}, actual {actual_count}"

    pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
    for match in re.finditer(pattern, svg_code):
        b64_data = match.group(1)
        if len(b64_data) % 4 != 0:
            return False, f"Truncated base64 data found (length {len(b64_data)} is not a multiple of 4)"
        if len(b64_data) < 100:
            return False, f"Suspiciously short base64 data found (length {len(b64_data)}), possibly truncated"

    return True, f"base64 image validation passed: {actual_count} images"


def svg_to_png(svg_path: str, output_path: str, scale: float = 1.0) -> Optional[str]:
    """Convert SVG to PNG"""
    try:
        import cairosvg
        cairosvg.svg2png(url=svg_path, write_to=output_path, scale=scale)
        return output_path
    except ImportError:
        print("  Warning: cairosvg not installed, trying alternative methods")
        try:
            from svglib.svglib import svg2rlg
            from reportlab.graphics import renderPM
            drawing = svg2rlg(svg_path)
            renderPM.drawToFile(drawing, output_path, fmt="PNG")
            return output_path
        except ImportError:
            print("  Warning: svglib also not installed, cannot convert SVG to PNG")
            return None
        except Exception as e:
            print(f"  Warning: svglib conversion failed: {e}")
            return None
    except Exception as e:
        print(f"  Warning: cairosvg conversion failed: {e}")
        return None


def optimize_svg_with_llm(
    figure_path: str,
    samed_path: str,
    final_svg_path: str,
    output_path: str,
    api_key: str,
    model: str,
    base_url: str,
    provider: ProviderType,
    max_iterations: int = 2,
    skip_base64_validation: bool = False,
    no_icon_mode: bool = False,
) -> str:
    """
    Optimize SVG using LLM to better align with the original image

    Args:
        figure_path: original image path
        samed_path: labeled image path
        final_svg_path: input SVG path
        output_path: output SVG path
        api_key: API key
        model: model name
        base_url: API base URL
        provider: API provider
        max_iterations: maximum iterations (0 = skip optimization)
        skip_base64_validation: whether to skip base64 image validation

    Returns:
        path to optimized SVG
    """
    print("\n" + "=" * 60)
    print("Step 4.6: LLM SVG optimization (position and style alignment)")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Max iterations: {max_iterations}")
    if no_icon_mode:
        print("No-icon mode: placeholder boxes are forbidden during optimization")

    # If iterations=0, copy file directly and skip optimization
    if max_iterations == 0:
        print("  Iterations = 0, skipping LLM optimization")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(final_svg_path, output_path)
        print(f"  Copying template directly: {final_svg_path} -> {output_path}")
        return str(output_path)

    with open(final_svg_path, 'r', encoding='utf-8') as f:
        current_svg = f.read()

    output_dir = Path(final_svg_path).parent

    original_image_count = 0
    if not skip_base64_validation:
        original_image_count = count_base64_images(current_svg)
        print(f"Original SVG contains {original_image_count} embedded images")
    else:
        print("Skipping base64 image validation (template SVG)")

    for iteration in range(max_iterations):
        print(f"\n  Optimization iteration {iteration + 1}/{max_iterations}")
        print("  " + "-" * 50)

        current_svg_path = output_dir / f"temp_svg_iter_{iteration}.svg"
        current_png_path = output_dir / f"temp_png_iter_{iteration}.png"

        with open(current_svg_path, 'w', encoding='utf-8') as f:
            f.write(current_svg)

        png_result = svg_to_png(str(current_svg_path), str(current_png_path))

        if png_result is None:
            print("  Cannot convert SVG to PNG, skipping optimization")
            break

        figure_img = Image.open(figure_path)
        samed_img = Image.open(samed_path)
        current_png_img = Image.open(str(current_png_path))

        if no_icon_mode:
            prompt = f"""You are an expert SVG optimizer. Compare the current SVG rendering with the original figure and optimize the SVG code to better match the original.

I'm providing you with 4 inputs:
1. **Image 1 (figure.png)**: The original target figure that we want to replicate
2. **Image 2 (samed.png)**: The SAM reference image for this run. No valid icon boxes were detected.
3. **Image 3 (current SVG rendered as PNG)**: The current state of our SVG
4. **Current SVG code**: The SVG code that needs optimization

Please carefully compare and optimize:
1. Overall layout and spatial alignment
2. Text positions, font sizes, and colors
3. Arrows, connectors, borders, and strokes
4. Shapes, grouping, and visual hierarchy

**CURRENT SVG CODE:**
```xml
{current_svg}
```

**IMPORTANT:**
- Output ONLY the optimized SVG code
- Start with <svg and end with </svg>
- Do NOT include markdown formatting or explanations
- No valid icon placeholders exist for this figure
- Do NOT add gray rectangles, AF labels, placeholder groups, or synthetic icon boxes
- Focus on position and style corrections"""
        else:
            prompt = f"""You are an expert SVG optimizer. Compare the current SVG rendering with the original figure and optimize the SVG code to better match the original.

I'm providing you with 4 inputs:
1. **Image 1 (figure.png)**: The original target figure that we want to replicate
2. **Image 2 (samed.png)**: The same figure with icon positions marked as gray rectangles with labels (<AF>01, <AF>02, etc.)
3. **Image 3 (current SVG rendered as PNG)**: The current state of our SVG
4. **Current SVG code**: The SVG code that needs optimization

Please carefully compare and check the following **TWO MAJOR ASPECTS with EIGHT KEY POINTS**:

## ASPECT 1: POSITION
1. **Icons**: Are icon placeholder positions matching the original?
2. **Text**: Are text elements positioned correctly?
3. **Arrows**: Are arrows starting/ending at correct positions?
4. **Lines/Borders**: Are lines and borders aligned properly?

## ASPECT 2: STYLE
5. **Icons**: Icon placeholder sizes, proportions (must have gray fill #808080, black border, and centered label)
6. **Text**: Font sizes, colors, weights
7. **Arrows**: Arrow styles, thicknesses, colors
8. **Lines/Borders**: Line styles, colors, stroke widths

**CURRENT SVG CODE:**
```xml
{current_svg}
```

**IMPORTANT:**
- Output ONLY the optimized SVG code
- Start with <svg and end with </svg>
- Do NOT include markdown formatting or explanations
- Keep all icon placeholder structures intact (the <g> elements with id like "AF01")
- Focus on position and style corrections"""

        contents = [prompt, figure_img, samed_img, current_png_img]

        try:
            print("  Sending optimization request...")
            content = call_llm_multimodal(
                contents=contents,
                api_key=api_key,
                model=model,
                base_url=base_url,
                provider=provider,
                max_tokens=50000,
                temperature=0.3,
            )

            if not content:
                print("  Response is empty")
                continue

            optimized_svg = extract_svg_code(content)

            if not optimized_svg:
                print("  Could not extract SVG code from response")
                continue

            is_valid, errors = validate_svg_syntax(optimized_svg)

            if not is_valid:
                print(f"  Optimized SVG has syntax errors, attempting repair...")
                optimized_svg = fix_svg_with_llm(
                    svg_code=optimized_svg,
                    errors=errors,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                    provider=provider,
                )

            if not skip_base64_validation:
                images_valid, images_msg = validate_base64_images(optimized_svg, original_image_count)
                if not images_valid:
                    print(f"  Warning: {images_msg}")
                    print("  Rejecting this optimization, keeping previous SVG version")
                    continue
                print(f"  {images_msg}")

            current_svg = optimized_svg
            print("  Optimization iteration complete")

        except Exception as e:
            print(f"  Error during optimization: {e}")
            continue

        try:
            current_svg_path.unlink(missing_ok=True)
            current_png_path.unlink(missing_ok=True)
        except:
            pass

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(current_svg)

    final_png_path = output_path.with_suffix('.png')
    svg_to_png(str(output_path), str(final_png_path))
    print(f"\n  Optimized SVG saved: {output_path}")
    print(f"  PNG preview saved: {final_png_path}")

    return str(output_path)


# ============================================================================
# Main function: full pipeline
# ============================================================================

def method_to_svg(
    method_text: str,
    output_dir: str = "./output",
    api_key: str = None,
    base_url: str = None,
    provider: ProviderType = "bianxie",
    image_gen_model: str = None,
    svg_gen_model: str = None,
    sam_prompts: str = "icon",
    min_score: float = 0.5,
    sam_backend: Literal["local", "fal", "roboflow", "api"] = "local",
    sam_api_key: Optional[str] = None,
    sam_max_masks: int = 32,
    rmbg_model_path: Optional[str] = None,
    stop_after: int = 5,
    placeholder_mode: PlaceholderMode = "label",
    optimize_iterations: int = 2,
    merge_threshold: float = 0.9,
    image_size: str = GEMINI_DEFAULT_IMAGE_SIZE,
) -> dict:
    """
    Full pipeline: Paper Method → SVG with Icons

    Args:
        method_text: Paper method text content
        output_dir: output directory
        api_key: API key
        base_url: API base URL
        provider: API provider
        image_gen_model: image generation model
        svg_gen_model: SVG generation model
        sam_prompts: SAM3 text prompts, supports comma-separated multiple prompts (e.g. "icon,diagram,arrow")
        min_score: SAM3 minimum confidence
        sam_backend: SAM3 backend (local/fal/roboflow/api)
        sam_api_key: SAM3 API key (for api mode)
        sam_max_masks: SAM3 API max masks count (for api mode)
        rmbg_model_path: RMBG model path
        stop_after: stop after specified step
        placeholder_mode: placeholder mode
            - "none": no special style
            - "box": pass boxlib coordinates
            - "label": gray fill+black border+numbered labels (recommended)
        optimize_iterations: step 4.6 optimization iteration count (0 = skip)
        merge_threshold: box merge threshold, boxes with overlap above this value are merged (0=no merge, default 0.9)

    Returns:
        result dict
    """
    if not api_key:
        raise ValueError("api_key is required")

    # Get default configuration
    config = PROVIDER_CONFIGS[provider]
    if base_url is None:
        base_url = config["base_url"]
    if image_gen_model is None:
        image_gen_model = config["default_image_model"]
    if svg_gen_model is None:
        svg_gen_model = config["default_svg_model"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("Paper Method to SVG icon replacement pipeline (Label mode enhanced + Box merge)")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"Output directory: {output_dir}")
    print(f"Image gen model: {image_gen_model}")
    print(f"SVG model: {svg_gen_model}")
    print(f"SAM prompts: {sam_prompts}")
    print(f"Min confidence: {min_score}")
    sam_backend_value = "fal" if sam_backend == "api" else sam_backend
    print(f"SAM backend: {sam_backend_value}")
    if sam_backend_value == "fal":
        print(f"SAM3 API max_masks: {sam_max_masks}")
    print(f"Run until step: {stop_after}")
    print(f"Placeholder mode: {placeholder_mode}")
    print(f"Optimization iterations: {optimize_iterations}")
    print(f"Box merge threshold: {merge_threshold}")
    if provider == "gemini":
        print(f"Image resolution: {image_size}")
    print("=" * 60)

    # Step 1: generate image
    figure_path = output_dir / "figure.png"
    generate_figure_from_method(
        method_text=method_text,
        output_path=str(figure_path),
        api_key=api_key,
        model=image_gen_model,
        base_url=base_url,
        provider=provider,
        image_size=image_size,
    )

    if stop_after == 1:
        print("\n" + "=" * 60)
        print("Stopped after step 1")
        print("=" * 60)
        return {
            "figure_path": str(figure_path),
            "samed_path": None,
            "boxlib_path": None,
            "icon_infos": [],
            "template_svg_path": None,
            "optimized_template_path": None,
            "final_svg_path": None,
        }

    # Step 2: SAM3 segmentation (includes box merge)
    samed_path, boxlib_path, valid_boxes = segment_with_sam3(
        image_path=str(figure_path),
        output_dir=str(output_dir),
        text_prompts=sam_prompts,
        min_score=min_score,
        merge_threshold=merge_threshold,
        sam_backend=sam_backend_value,
        sam_api_key=sam_api_key,
        sam_max_masks=sam_max_masks,
    )

    no_icon_mode = len(valid_boxes) == 0
    if no_icon_mode:
        print("\nWarning: no valid icons detected, switching to pure SVG fallback mode")
    else:
        print(f"\nDetected {len(valid_boxes)} icons")

    if stop_after == 2:
        print("\n" + "=" * 60)
        print("Stopped after step 2")
        print("=" * 60)
        return {
            "figure_path": str(figure_path),
            "samed_path": samed_path,
            "boxlib_path": boxlib_path,
            "icon_infos": [],
            "template_svg_path": None,
            "optimized_template_path": None,
            "final_svg_path": None,
        }

    # Step 3: crop + background removal
    icon_infos = []
    if no_icon_mode:
        print("Step 3 skipped: currently in no-icon fallback mode")
    else:
        _ensure_rmbg2_access_ready(rmbg_model_path)
        icon_infos = crop_and_remove_background(
            image_path=str(figure_path),
            boxlib_path=boxlib_path,
            output_dir=str(output_dir),
            rmbg_model_path=rmbg_model_path,
        )

    if stop_after == 3:
        print("\n" + "=" * 60)
        print("Stopped after step 3")
        print("=" * 60)
        return {
            "figure_path": str(figure_path),
            "samed_path": samed_path,
            "boxlib_path": boxlib_path,
            "icon_infos": icon_infos,
            "template_svg_path": None,
            "optimized_template_path": None,
            "final_svg_path": None,
        }

    # Step 4: generate SVG template
    template_svg_path = output_dir / "template.svg"
    optimized_template_path = output_dir / "optimized_template.svg"
    final_svg_path = output_dir / "final.svg"
    try:
        generate_svg_template(
            figure_path=str(figure_path),
            samed_path=samed_path,
            boxlib_path=boxlib_path,
            output_path=str(template_svg_path),
            api_key=api_key,
            model=svg_gen_model,
            base_url=base_url,
            provider=provider,
            placeholder_mode=placeholder_mode,
            no_icon_mode=no_icon_mode,
        )

        # Step 4.6: LLM optimize SVG template (iterations configurable, 0=skip)
        optimize_svg_with_llm(
            figure_path=str(figure_path),
            samed_path=samed_path,
            final_svg_path=str(template_svg_path),
            output_path=str(optimized_template_path),
            api_key=api_key,
            model=svg_gen_model,
            base_url=base_url,
            provider=provider,
            max_iterations=optimize_iterations,
            skip_base64_validation=True,
            no_icon_mode=no_icon_mode,
        )
    except Exception as exc:
        if not no_icon_mode:
            raise
        print(f"SVG rebuild failed in no-icon mode ({exc}), using fallback SVG with embedded original image")
        create_embedded_figure_svg(
            figure_path=str(figure_path),
            output_path=str(final_svg_path),
        )

    if stop_after == 4:
        print("\n" + "=" * 60)
        print("Stopped after step 4")
        print("=" * 60)
        return {
            "figure_path": str(figure_path),
            "samed_path": samed_path,
            "boxlib_path": boxlib_path,
            "icon_infos": icon_infos,
            "template_svg_path": str(template_svg_path) if template_svg_path.is_file() else None,
            "optimized_template_path": str(optimized_template_path) if optimized_template_path.is_file() else None,
            "final_svg_path": None,
        }

    svg_template_for_replace = optimized_template_path if optimized_template_path.is_file() else template_svg_path

    # Step 5: icon replacement
    if no_icon_mode:
        if svg_template_for_replace.is_file():
            shutil.copyfile(svg_template_for_replace, final_svg_path)
            print("No-icon mode: skipping icon replacement, outputting SVG directly")
        else:
            print("No-icon mode: missing template SVG, generating fallback final.svg")
            create_embedded_figure_svg(
                figure_path=str(figure_path),
                output_path=str(final_svg_path),
            )
    else:
        # Step 4.7: Coordinate system alignment
        print("\n" + "-" * 50)
        print("Step 4.7: Coordinate system alignment")
        print("-" * 50)

        figure_img = Image.open(figure_path)
        figure_width, figure_height = figure_img.size
        print(f"Original image size: {figure_width} x {figure_height}")

        with open(svg_template_for_replace, 'r', encoding='utf-8') as f:
            svg_code = f.read()

        svg_width, svg_height = get_svg_dimensions(svg_code)

        if svg_width and svg_height:
            print(f"SVG size: {svg_width} x {svg_height}")

            if abs(svg_width - figure_width) < 1 and abs(svg_height - figure_height) < 1:
                print("Sizes match, using 1:1 coordinate mapping")
                scale_factors = (1.0, 1.0)
            else:
                scale_x, scale_y = calculate_scale_factors(
                    figure_width, figure_height, svg_width, svg_height
                )
                scale_factors = (scale_x, scale_y)
                print(f"Size mismatch, computing scale factors: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")
        else:
            print("Warning: cannot extract SVG dimensions, using 1:1 coordinate mapping")
            scale_factors = (1.0, 1.0)

        replace_icons_in_svg(
            template_svg_path=str(svg_template_for_replace),
            icon_infos=icon_infos,
            output_path=str(final_svg_path),
            scale_factors=scale_factors,
            match_by_label=(placeholder_mode == "label"),
        )

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"Original image: {figure_path}")
    print(f"Labeled image: {samed_path}")
    print(f"Box info: {boxlib_path}")
    print(f"Icon count: {len(icon_infos)}")
    print(f"SVG template: {template_svg_path}")
    print(f"Optimized template: {optimized_template_path}")
    print(f"Final SVG: {final_svg_path}")

    return {
        "figure_path": str(figure_path),
        "samed_path": samed_path,
        "boxlib_path": boxlib_path,
        "icon_infos": icon_infos,
        "template_svg_path": str(template_svg_path) if template_svg_path.is_file() else None,
        "optimized_template_path": str(optimized_template_path) if optimized_template_path.is_file() else None,
        "final_svg_path": str(final_svg_path),
    }


def create_embedded_figure_svg(
    figure_path: str,
    output_path: str,
) -> str:
    """Wrap the generated raster figure in a minimal SVG as a final fallback."""
    figure_img = Image.open(figure_path)
    width, height = figure_img.size
    buf = io.BytesIO()
    figure_img.save(buf, format="PNG")
    figure_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    svg_code = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'  <image x="0" y="0" width="{width}" height="{height}" '
        f'href="data:image/png;base64,{figure_b64}" preserveAspectRatio="none"/>\n'
        f"</svg>\n"
    )

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, 'w', encoding='utf-8') as f:
        f.write(svg_code)

    print(f"Fallback SVG with embedded figure.png saved: {output_path_obj}")
    return str(output_path_obj)


# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paper Method to SVG icon replacement tool (Label mode enhanced + Box merge)"
    )

    # Input parameters
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--method_text", help="Paper method text content")
    input_group.add_argument("--method_file", default="./paper.txt", help="Path to text file containing the paper method")

    # Output parameters
    parser.add_argument("--output_dir", default="./output", help="Output directory (default: ./output)")

    # Provider parameters
    parser.add_argument(
        "--provider",
        choices=["openrouter", "bianxie", "gemini"],
        default="bianxie",
        help="API provider (default: bianxie)"
    )

    # API parameters
    parser.add_argument("--api_key", default=None, help="API Key")
    parser.add_argument("--base_url", default=None, help="API base URL (auto-set based on provider by default)")

    # Model parameters
    parser.add_argument("--image_model", default=None, help="Image generation model (auto-set based on provider by default)")
    parser.add_argument(
        "--image_size",
        choices=list(IMAGE_SIZE_CHOICES),
        default=GEMINI_DEFAULT_IMAGE_SIZE,
        help="Image resolution (options: 1K/2K/4K, default: 4K)",
    )
    parser.add_argument("--svg_model", default=None, help="SVG generation model (auto-set based on provider by default)")

    # Step 1 reference image parameters
    parser.add_argument(
        "--use_reference_image",
        action="store_true",
        help="Step 1: use reference image style (requires --reference_image_path)",
    )
    parser.add_argument("--reference_image_path", default=None, help="Reference image path (optional)")

    # SAM3 parameters
    parser.add_argument("--sam_prompt", default="icon,robot,animal,person", help="SAM3 text prompts, supports comma-separated multiple prompts (e.g. 'icon,diagram,arrow', default: icon)")
    parser.add_argument("--min_score", type=float, default=0.0, help="SAM3 minimum confidence threshold (default: 0.0)")
    parser.add_argument(
        "--sam_backend",
        choices=["local", "fal", "roboflow", "api"],
        default="local",
        help="SAM3 backend: local(local deploy)/fal(fal.ai)/roboflow(Roboflow)/api(legacy alias=fal)",
    )
    parser.add_argument("--sam_api_key", default=None, help="SAM3 API key (defaults to FAL_KEY)")
    parser.add_argument(
        "--sam_max_masks",
        type=int,
        default=32,
        help="SAM3 API max masks count (api backend only, default: 32)",
    )

    # RMBG parameters
    parser.add_argument("--rmbg_model_path", default=None, help="RMBG model local path (optional)")

    # Pipeline control parameters
    parser.add_argument(
        "--stop_after",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=5,
        help="Stop after specified step (1-5, default: 5 for full pipeline)"
    )

    # Placeholder mode parameters
    parser.add_argument(
        "--placeholder_mode",
        choices=["none", "box", "label"],
        default="label",
        help="Placeholder mode: none(no style)/box(pass coords)/label(label matching) (default: label)"
    )

    # Step 4.6 optimization iteration count parameter
    parser.add_argument(
        "--optimize_iterations",
        type=int,
        default=0,
        help="Step 4.6 LLM optimization iterations (0 = skip, default: 0)"
    )

    # Box merge threshold parameter
    parser.add_argument(
        "--merge_threshold",
        type=float,
        default=0.001,
        help="Box merge threshold, boxes with overlap above this value are merged (0=no merge, default: 0.9)"
    )

    args = parser.parse_args()

    if args.use_reference_image and not args.reference_image_path:
        parser.error("--use_reference_image requires --reference_image_path")
    if args.reference_image_path and not Path(args.reference_image_path).is_file():
        parser.error(f"Reference image does not exist: {args.reference_image_path}")

    USE_REFERENCE_IMAGE = bool(args.use_reference_image)
    REFERENCE_IMAGE_PATH = args.reference_image_path
    if REFERENCE_IMAGE_PATH:
        USE_REFERENCE_IMAGE = True

    # Get method text: prefer --method_text
    method_text = args.method_text
    if method_text is None:
        with open(args.method_file, 'r', encoding='utf-8') as f:
            method_text = f.read()

    # Run full pipeline
    result = method_to_svg(
        method_text=method_text,
        output_dir=args.output_dir,
        api_key=args.api_key,
        base_url=args.base_url,
        provider=args.provider,
        image_gen_model=args.image_model,
        image_size=args.image_size,
        svg_gen_model=args.svg_model,
        sam_prompts=args.sam_prompt,
        min_score=args.min_score,
        sam_backend=args.sam_backend,
        sam_api_key=args.sam_api_key,
        sam_max_masks=args.sam_max_masks,
        rmbg_model_path=args.rmbg_model_path,
        stop_after=args.stop_after,
        placeholder_mode=args.placeholder_mode,
        optimize_iterations=args.optimize_iterations,
        merge_threshold=args.merge_threshold,
    )
