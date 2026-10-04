import sys
from pathlib import Path

# The lesson agents sit in agents/ next to this script, and helpers.py lives at
# the repo root one level up, so put both on the path no matter how the script
# is launched.
LESSONS_DIR = Path(__file__).resolve().parent
for import_root in (LESSONS_DIR.parent, LESSONS_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import litellm
from IPython import get_ipython
from IPython.display import Markdown, display

from agents.policy_agents import PolicyAgent

# Keep the "Give Feedback / Get Help" banners out of the output when litellm
# retries or falls back to another model.
litellm.suppress_debug_info = True

prompt = "How much would I pay for mental health therapy?"

# PolicyAgent loads the environment and the policy PDF, then answers the
# question against it.
policy_agent = PolicyAgent()
response_text = policy_agent.answer_query(prompt)

if get_ipython() is not None:
    # In a notebook, render as Markdown; "$" is already escaped by the agent so
    # it is not read as LaTeX.
    display(Markdown(response_text))
else:
    print(response_text)
