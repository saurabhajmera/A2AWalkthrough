import base64
from pathlib import Path 

import litellm
from IPython import get_ipython
from IPython.display import Markdown, display

from helpers import setup_env

setup_env()

# Keep the "Give Feedback / Get Help" banners out of the output when litellm
# retries or falls back to another model.
litellm.suppress_debug_info = True

with Path("data/2026AnthemgHIPSBC.pdf").open("rb") as file:
    pdf_data = base64.standard_b64encode(file.read()).decode("utf-8")

prompt = "How much would I pay for mental health therapy?"

response = litellm.completion(
    model="gemini/gemini-3-flash-preview",
    # For Vertex AI
    # model = "vertex_ai/gemini-3-flash-preview",
    reasoning_effort="minimal",
    max_tokens=1000,
    # The flash models return intermittent 503s under load; litellm does not
    # retry by default, so back off and retry, then fall back to other models.
    num_retries=3,
    timeout=120,
    fallbacks=["gemini/gemini-3.5-flash", "gemini/gemini-3.8-flash"],
    messages=[
        {
            "role":"system",
            "content":"You are an expert insurance agent designed to assist with coverage queries. Use the provided documents to answer questions about insurance policies.If the information is not available in the documents, respond with 'I don't know'",

        },
           {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:application/pdf;base64,{pdf_data}"},
                },
            ],
        },


    ],
)

response_text = response.choices[0].message.content

if get_ipython() is not None:
    # In a notebook, render as Markdown; escape "$" so it is not read as LaTeX.
    display(Markdown(response_text.replace("$", r"\$")))
else:
    print(response_text)