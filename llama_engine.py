import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MODEL_NAME = "llama3"   # change if using another model


def generate_answer(context, question):
    """
    Generates a concise answer using LLaMA via Ollama.
    The model is forced to answer ONLY from the retrieved context.
    """

    prompt = f"""
You are an official AI assistant for NIT Jalandhar.

STRICT INSTRUCTIONS:
- Use ONLY the information provided in the Context.
- Do NOT use outside knowledge.
- If the answer is not present in the Context, reply exactly:
  "I could not find this information in official records."
- Provide a clear and concise answer.
- If dates, numbers, or names are asked, extract them precisely.

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.2  # lower = more factual, less hallucination
            },
            timeout=(2.0, 30.0)
        )

        response.raise_for_status()
        result = response.json()

        answer = result.get("response", "").strip()

        answer = "\n".join(line.strip() for line in answer.splitlines() if line.strip())

        return answer if answer else "I could not generate a response."

    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return f"Based on NIT Jalandhar records:\n\n{context}\n\n*(Note: Ollama is offline or timed out. Start Ollama with 'ollama run llama3' for AI-synthesized answers.)*"

    except Exception as e:
        return f"Based on NIT Jalandhar records:\n\n{context}\n\n*(Note: {str(e)})*"
