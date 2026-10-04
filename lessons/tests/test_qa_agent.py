"""Regression test for lessons/qa_agent.py.

Runs the lesson script with litellm.completion stubbed out (no network calls
and no API key needed) and asserts the observable behaviour: the policy PDF is
attached, the question is asked, the insurance system prompt is used, and the
model's answer reaches stdout.

Run from the repo root:

    .venv/bin/python lessons/tests/test_qa_agent.py
"""

import base64
import contextlib
import io
import runpy
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO_ROOT / "lessons" / "qa_agent.py"
PDF = REPO_ROOT / "data" / "2026AnthemgHIPSBC.pdf"

ANSWER = "In-network therapy costs $25 per visit after the deductible."

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _fake_response(text: str) -> mock.Mock:
    """Builds the minimal litellm response shape the lesson code reads."""
    message = mock.Mock()
    message.content = text
    choice = mock.Mock()
    choice.message = message
    response = mock.Mock()
    response.choices = [choice]
    return response


def _flatten(value: object, out: list[str]) -> None:
    """Collects every string found anywhere inside a nested message payload."""
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _flatten(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _flatten(item, out)


def _unescape_dollars(text: str) -> str:
    """Drops the LaTeX-safe escaping so outputs compare equal either way."""
    return text.replace(r"\$", "$")


class QaAgentScriptTest(unittest.TestCase):
    """Executes qa_agent.py once per test with the model call intercepted."""

    def setUp(self) -> None:
        import litellm

        self.calls: list[dict] = []

        def spy(*args: object, **kwargs: object) -> mock.Mock:
            self.calls.append(kwargs)
            return _fake_response(ANSWER)

        patcher = mock.patch.object(litellm, "completion", side_effect=spy)
        patcher.start()
        self.addCleanup(patcher.stop)

        # runpy always re-executes the script, but keep sys.modules clean so a
        # cached agent module cannot mask an import regression.
        for name in ("qa_agent", "agents", "agents.policy_agents", "policy_agent"):
            sys.modules.pop(name, None)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            runpy.run_path(str(SCRIPT), run_name="__main__")
        self.output = stdout.getvalue()

    def test_calls_the_model_exactly_once(self) -> None:
        self.assertEqual(len(self.calls), 1)

    def test_uses_insurance_system_prompt(self) -> None:
        messages = self.calls[0]["messages"]
        system = [m for m in messages if m["role"] == "system"]
        self.assertEqual(len(system), 1)
        content = system[0]["content"]
        self.assertIn("insurance", content.lower())
        self.assertIn("I don't know", content)

    def test_sends_the_mental_health_question(self) -> None:
        user = [m for m in self.calls[0]["messages"] if m["role"] == "user"]
        self.assertEqual(len(user), 1)
        strings: list[str] = []
        _flatten(user[0]["content"], strings)
        self.assertTrue(
            any("mental health therapy" in s for s in strings),
            f"question not found in user message: {strings[:2]}",
        )

    def test_attaches_the_policy_pdf_as_base64(self) -> None:
        expected = base64.standard_b64encode(PDF.read_bytes()).decode("utf-8")
        strings: list[str] = []
        _flatten(self.calls[0]["messages"], strings)
        urls = [s for s in strings if s.startswith("data:application/pdf;base64,")]
        self.assertEqual(len(urls), 1, "expected exactly one inline PDF")
        self.assertEqual(urls[0], f"data:application/pdf;base64,{expected}")

    def test_prints_the_answer(self) -> None:
        self.assertIn(_unescape_dollars(ANSWER), _unescape_dollars(self.output))


if __name__ == "__main__":
    unittest.main(verbosity=2)
