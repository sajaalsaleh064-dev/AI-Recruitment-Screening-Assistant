"""Step 7a - Talk to a local LLM through Ollama.

Ollama runs language models on your own computer. Nothing is sent to the
internet, which matters here because CVs are personal data.

This module has one job: send a prompt and get back a Python dictionary.
The prompts themselves live in ai_extraction.py.

Check the connection:  python -m src.ollama_client
"""
import json
import logging

import ollama

from src.config import OLLAMA_HOST, OLLAMA_MODEL

logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Raised when Ollama is unreachable, the model is missing or the
    answer is not usable JSON."""


def _client():
    # A long timeout: the first request also loads the model into memory.
    return ollama.Client(host=OLLAMA_HOST, timeout=600)


def check_ollama(model=OLLAMA_MODEL):
    """Make sure the Ollama server is running and the model is downloaded."""
    try:
        installed = [m.model for m in _client().list().models]
    except Exception as e:  # connection refused, timeout, ...
        raise OllamaError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. Start the Ollama app "
            f"(or run 'ollama serve') and try again. Details: {e}"
        ) from e
    # "llama3.1" should match the installed name "llama3.1:latest".
    wanted = model if ":" in model else f"{model}:latest"
    if wanted not in installed:
        raise OllamaError(
            f"Model '{model}' is not downloaded. Run: ollama pull {model}\n"
            f"Installed models: {', '.join(installed) or 'none'}"
        )
    return True


def parse_json_object(text):
    """Turn the model's answer into a dictionary.

    Models sometimes wrap JSON in ```json fences or add a sentence around
    it, so we keep only the part from the first "{" to the last "}".
    """
    if not isinstance(text, str) or not text.strip():
        raise OllamaError("The model returned an empty answer")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise OllamaError("The model's answer contains no JSON object")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise OllamaError(f"The model returned invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise OllamaError("The model's JSON is not an object")
    return data


def _chat(system_prompt, user_prompt, schema, model):
    """One request to the model. Returns the raw answer text."""
    try:
        response = _client().chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # A JSON schema here forces the answer to follow that structure.
            format=schema or "json",
            options={
                "temperature": 0,  # no creativity: same input -> same output
                "num_ctx": 8192,   # room for a full CV plus the instructions
            },
        )
    except ollama.ResponseError as e:
        raise OllamaError(f"Ollama returned an error: {e.error}") from e
    except Exception as e:
        raise OllamaError(f"Could not get an answer from Ollama: {e}") from e
    return response.message.content


def generate_json(system_prompt, user_prompt, schema=None, model=OLLAMA_MODEL,
                  retries=1, chat=_chat):
    """Ask the model a question and return its answer as a dictionary.

    If the answer is not valid JSON the request is repeated (`retries`
    more times) before giving up with an OllamaError.
    `chat` can be replaced in tests so no real model is needed.
    """
    last_error = None
    for attempt in range(1, retries + 2):
        answer = chat(system_prompt, user_prompt, schema, model)
        try:
            return parse_json_object(answer)
        except OllamaError as e:
            last_error = e
            logger.warning("Attempt %d: %s", attempt, e)
    raise OllamaError(f"No valid JSON after {retries + 1} attempt(s): {last_error}")


if __name__ == "__main__":
    try:
        check_ollama()
        print(f"[ok] Ollama is running and model '{OLLAMA_MODEL}' is available")
        reply = generate_json(
            "You answer only with JSON.",
            'Return {"status": "ready", "capital_of_jordan": <the city>}.',
        )
        print(f"[ok] Test answer: {reply}")
    except OllamaError as e:
        print(f"[FAIL] {e}")
