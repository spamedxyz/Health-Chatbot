"""
Task 4: General Health Query Chatbot.

Prompt-engineered health information assistant with simple safety filters.
It can use OpenAI when OPENAI_API_KEY is configured, and otherwise falls back
to local educational responses so the project works offline.
"""

from __future__ import annotations

import os
import re
import sys

SYSTEM_PROMPT = """You are a helpful, friendly medical information assistant.

Guidelines:
- Provide clear, general health information in plain language.
- Be empathetic and supportive in tone.
- Do not diagnose conditions or prescribe specific treatments.
- Do not recommend stopping or changing prescribed medicines.
- For emergencies, advise contacting local emergency services immediately.
- If unsure, say so and recommend speaking with a qualified clinician.
- End every response with the required safety disclaimer.
"""

DISCLAIMER = (
    "\n\nDisclaimer: This is general information only, not medical advice. "
    "Please consult a qualified healthcare professional for personal medical concerns."
)

UNSAFE_PATTERNS = [
    r"\bhow\s+to\s+(make|synthesize|create)\s+(drugs|narcotics|poison)\b",
    r"\boverdose\b",
    r"\bself[- ]?harm\b",
    r"\bsuicide\b",
    r"\bkill\s+myself\b",
    r"\bignore\s+(doctor|physician)\b",
    r"\bstop\s+taking\s+(my\s+)?medication\b",
    r"\bchange\s+(my\s+)?dose\b",
]

OFFLINE_RESPONSES = {
    "sore throat": (
        "A sore throat is often caused by a viral infection such as a cold or flu. "
        "Other possible causes include allergies, dry air, acid reflux, voice strain, "
        "or a bacterial infection. Rest, fluids, warm drinks, and salt-water gargles "
        "may help mild symptoms. Seek urgent care for trouble breathing, severe "
        "swelling, dehydration, very high fever, or symptoms that do not improve."
    ),
    "paracetamol": (
        "Paracetamol, also called acetaminophen, can be safe for children when the "
        "dose is based on the child's weight and the product label or a clinician's "
        "instructions are followed. Avoid giving multiple medicines that contain "
        "paracetamol at the same time because too much can harm the liver. Ask a "
        "doctor or pharmacist for infants, chronic illness, or uncertain dosing."
    ),
    "headache": (
        "Headaches can come from dehydration, lack of sleep, stress, eye strain, "
        "sinus congestion, migraine, or other causes. Resting, drinking water, and "
        "avoiding triggers may help mild headaches. Sudden severe headache, weakness, "
        "confusion, fever with stiff neck, head injury, or vision changes need urgent care."
    ),
}


def is_unsafe_query(query: str) -> bool:
    """Return True when the query asks for unsafe or inappropriate guidance."""
    query_lower = query.lower()
    return any(re.search(pattern, query_lower) for pattern in UNSAFE_PATTERNS)


def add_disclaimer(response: str) -> str:
    """Append the required medical disclaimer if it is missing."""
    if DISCLAIMER.strip() not in response:
        return response.rstrip() + DISCLAIMER
    return response


def get_refusal_message() -> str:
    """Return a safe refusal for dangerous or crisis-related requests."""
    return add_disclaimer(
        "I cannot help with that request. If you or someone else may be in immediate "
        "danger, contact local emergency services now. If this is about self-harm or "
        "suicide, please reach out to a crisis helpline or a trusted person right away."
    )


def query_openai(user_query: str, api_key: str | None = None) -> str:
    """Send a health-information query to the OpenAI Chat Completions API."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        max_tokens=450,
        temperature=0.4,
    )
    return response.choices[0].message.content or ""


def query_offline_fallback(user_query: str) -> str:
    """Return a safe local response when no API key is available."""
    query_lower = user_query.lower()
    for keyword, response in OFFLINE_RESPONSES.items():
        if keyword in query_lower:
            return response

    return (
        "I can share general health education, but I cannot diagnose symptoms or "
        "create a personal treatment plan. For reliable information, use sources "
        "such as WHO, CDC, NHS, or a qualified healthcare professional who can "
        "review the full situation."
    )


def chat(user_query: str, backend: str = "auto") -> str:
    """
    Answer a health query with safety checks.

    backend can be "auto", "openai", or "offline".
    """
    if not user_query.strip():
        return add_disclaimer("Please ask a general health question.")

    if is_unsafe_query(user_query):
        return get_refusal_message()

    try:
        if backend == "openai" or (backend == "auto" and os.environ.get("OPENAI_API_KEY")):
            response = query_openai(user_query)
        else:
            response = query_offline_fallback(user_query)
    except Exception as exc:
        response = (
            f"The online model was unavailable ({exc}). Here is a safer fallback: "
            + query_offline_fallback(user_query)
        )

    return add_disclaimer(response)


def main() -> None:
    """Run a small command-line chatbot demo."""
    print("=" * 60)
    print("Health Information Assistant - Educational Demo")
    print("Type 'quit' to exit")
    print("=" * 60)

    backend = "auto"
    if "--offline" in sys.argv:
        backend = "offline"
    elif "--openai" in sys.argv:
        backend = "openai"

    print("\nExample queries:")
    print("  - What causes a sore throat?")
    print("  - Is paracetamol safe for children?\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye. Take care.")
            break
        print(f"\nAssistant: {chat(user_input, backend=backend)}\n")


if __name__ == "__main__":
    main()
