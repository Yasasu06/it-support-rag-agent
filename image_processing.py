"""
Multi-modal input: extract the IT issue described in an uploaded screenshot.

The extracted text is treated exactly like a typed question — it flows through
the SAME existing RAG pipeline (including PII masking in run_agent_pipeline),
with no new retrieval logic or agents. This module only turns an image into
text for that pipeline to consume.
"""

import base64
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

SUPPORTED_IMAGE_TYPES = [
    "image/png", "image/jpeg", "image/jpg", "image/webp"
]

MAX_IMAGE_SIZE_MB = 10


def validate_image(
    file_bytes: bytes,
    content_type: str
) -> tuple[bool, Optional[str]]:
    """
    Validates an uploaded image before processing.
    Returns (is_valid, error_message).
    """
    if content_type not in SUPPORTED_IMAGE_TYPES:
        return False, (
            f"Unsupported image type: {content_type}. "
            f"Please upload PNG, JPEG, or WebP."
        )

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        return False, (
            f"Image too large ({size_mb:.1f}MB). "
            f"Maximum size is {MAX_IMAGE_SIZE_MB}MB."
        )

    if len(file_bytes) < 100:
        return False, "Image file appears to be empty or corrupted."

    return True, None


def extract_text_from_image(
    file_bytes: bytes,
    content_type: str,
    user_context: str = ""
) -> dict:
    """
    Uses GPT-4o-mini's vision capability to extract the
    IT issue described in an uploaded screenshot/image.
    Returns a dict with extracted_text and success flag -
    never raises, always returns a safe result even on
    failure (consistent with the fail-safe pattern used
    throughout this project).
    """
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        base64_image = base64.b64encode(
            file_bytes
        ).decode("utf-8")

        prompt_text = (
            "This is a screenshot from an IT support "
            "context (error message, application screen, "
            "or system issue). Describe the specific "
            "technical problem shown, including any exact "
            "error messages, error codes, or system state "
            "visible. Be concise and factual - only "
            "describe what is actually visible, do not "
            "guess at causes or solutions."
        )

        if user_context:
            prompt_text += (
                f"\n\nAdditional context from the user: "
                f"{user_context}"
            )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:{content_type};"
                                    f"base64,{base64_image}"
                                )
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
            timeout=20
        )

        extracted = response.choices[0].message.content

        return {
            "success": True,
            "extracted_text": extracted,
            "error": None
        }

    except Exception as e:
        return {
            "success": False,
            "extracted_text": "",
            "error": (
                f"Could not process image: {str(e)}. "
                f"Please describe your issue in text "
                f"instead."
            )
        }


def build_combined_query(
    typed_question: str,
    extracted_image_text: str
) -> str:
    """
    Combines typed question and image-extracted content
    into a single query for the existing RAG pipeline.
    Handles the case where only one input type is
    provided.
    """
    typed_question = (typed_question or "").strip()
    extracted_image_text = (
        extracted_image_text or ""
    ).strip()

    if typed_question and extracted_image_text:
        return (
            f"{typed_question}\n\n"
            f"[From uploaded screenshot: "
            f"{extracted_image_text}]"
        )
    elif extracted_image_text:
        return extracted_image_text
    else:
        return typed_question
