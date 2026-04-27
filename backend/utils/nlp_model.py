from __future__ import annotations

import logging
import re
import requests

logger = logging.getLogger(__name__)

# Model handles — populated lazily on first use so the backend starts
# even when torch/transformers are not yet installed.
_torch = None
_base_tokenizer = None
_base_model = None
_device = None
_model_load_attempted = False
_translation_models: dict = {}

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

TRANSLATION_MODEL_IDS = {
    "hindi":   "Helsinki-NLP/opus-mt-en-hi",
    "marathi": "Helsinki-NLP/opus-mt-en-mr",
    "tamil":   "Helsinki-NLP/opus-mt-en-ta",
    "telugu":  "Helsinki-NLP/opus-mt-en-te",
}


def _try_load_model() -> bool:
    """Attempt to load torch + Qwen model. Returns True on success."""
    global _torch, _base_tokenizer, _base_model, _device, _model_load_attempted
    if _model_load_attempted:
        return _base_model is not None
    _model_load_attempted = True
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        _torch = torch
        _device = torch.device("cpu")
        logger.info("Loading base NLM model: %s", BASE_MODEL_ID)
        _base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        _base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, torch_dtype=torch.float32
        )
        _base_model.to(_device).eval()
        logger.info("NLM model loaded successfully")
        return True
    except Exception as exc:
        logger.warning("NLM model unavailable (torch/transformers not ready): %s", exc)
        return False


def load_translation_model(lang: str):
    lang = lang.lower().strip()
    model_name = TRANSLATION_MODEL_IDS.get(lang)
    if not model_name:
        return None
    if lang not in _translation_models:
        try:
            from transformers import MarianTokenizer, MarianMTModel
            tok = MarianTokenizer.from_pretrained(model_name)
            mod = MarianMTModel.from_pretrained(model_name).to("cpu")
            _translation_models[lang] = (tok, mod)
        except Exception as exc:
            logger.warning("Translation model load failed (%s): %s", lang, exc)
            return None
    return _translation_models.get(lang)


def _clean_output(text: str) -> str:
    split = re.split(r"(?:Human|User|Assistant|System)\s*[:：]", text)
    return split[0].strip()


def _run_generation(prompt: str, max_tokens: int = 200) -> str:
    if not _try_load_model():
        return ""
    inputs = _base_tokenizer(prompt, return_tensors="pt", truncation=True).to(_device)
    prompt_len = inputs["input_ids"].shape[-1]
    with _torch.no_grad():
        output_ids = _base_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            eos_token_id=_base_tokenizer.eos_token_id,
            pad_token_id=_base_tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][prompt_len:]
    text = _base_tokenizer.decode(generated_ids, skip_special_tokens=True)
    return _clean_output(text)


def generate_nlm_reply(prompt: str, max_tokens: int = 200) -> str:
    if not _try_load_model():
        return "AI model is loading — please try again in a moment."
    try:
        prompt = (
            prompt
            .replace("<|system|>", "<|im_start|>system\n")
            .replace("<|user|>", "<|im_start|>user\n")
            .replace("<|assistant|>", "<|im_start|>assistant\n")
        )
        result = _run_generation(prompt, max_tokens)
        return result or "I'm sorry, I couldn't generate a response just now."
    except Exception as e:
        logger.error("Qwen generation error: %s", e)
        return "I'm sorry, I couldn't generate a response just now."


def generate_rag_reply(
    user_input: str,
    rag_context: str = "",
    memory_context: str = "",
    conversation_history: str = "",
    max_tokens: int = 250,
) -> str:
    if not _try_load_model():
        # Simple rule-based fallback when the model isn't loaded yet
        if rag_context:
            first_sentence = rag_context.split(".")[0].strip()
            if first_sentence:
                return first_sentence + "."
        return (
            "The AI model is still loading. "
            "For weather data, please use the Get Weather Probability section above."
        )
    messages = [
        {"role": "system", "content": (
            "You are a helpful AI assistant. "
            "When context is provided, answer using ONLY that context — do not add unsupported claims. "
            "Be direct and concise."
        )}
    ]
    if rag_context:
        messages[0]["content"] += "\n\nRelevant Knowledge: " + rag_context
    if memory_context:
        messages[0]["content"] += "\n\nConversation Memory: " + memory_context
    if conversation_history:
        messages[0]["content"] += "\n\nRecent Chat:\n" + conversation_history
    messages.append({"role": "user", "content": user_input})

    prompt = _base_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    try:
        result = _run_generation(prompt, max_tokens)
        return result or "I'm sorry, I couldn't generate a response just now."
    except Exception as e:
        logger.error("RAG generation error: %s", e)
        fallback = [
            {"role": "system", "content": "Give a short, correct answer."},
            {"role": "user", "content": user_input},
        ]
        fallback_prompt = _base_tokenizer.apply_chat_template(
            fallback, tokenize=False, add_generation_prompt=True
        )
        return generate_nlm_reply(fallback_prompt)


_LANG_CODES = {"hindi": "hi", "marathi": "mr", "tamil": "ta", "telugu": "te"}


def _mymemory_translate(phrase: str, lang_code: str) -> str | None:
    try:
        r = requests.get(
            "https://api.mymemory.translated.net/get",
            params={"q": phrase, "langpair": f"en|{lang_code}"},
            timeout=8,
        )
        data = r.json()
        result = data.get("responseData", {}).get("translatedText", "")
        if result and result.strip().lower() != phrase.strip().lower():
            return result.strip()
    except Exception as e:
        logger.warning("MyMemory translation failed: %s", e)
    return None


def translate_text(phrase: str, target_lang: str) -> str:
    if not target_lang:
        return "Please specify a target language (e.g., Hindi, Marathi)."
    lang = target_lang.lower().strip()
    lang_code = _LANG_CODES.get(lang)
    if not lang_code:
        return f"Sorry, translation to '{target_lang}' is not supported yet."

    result = _mymemory_translate(phrase, lang_code)
    if result:
        return result

    model_data = load_translation_model(lang)
    if not model_data:
        return "Translation failed. Please try again."
    tok, mod = model_data
    try:
        import torch
        batch = tok([phrase], return_tensors="pt", truncation=True)
        with torch.no_grad():
            generated = mod.generate(**batch, num_beams=4, early_stopping=True)
        out = tok.batch_decode(generated, skip_special_tokens=True)[0]
        return out.strip()
    except Exception as e:
        logger.error("MarianMT translation error: %s", e)
        return "Translation failed. Please try again."
