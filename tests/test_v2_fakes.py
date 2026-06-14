from types import SimpleNamespace
import json


class FakeGeminiClient:
    def __init__(self) -> None:
        self.created = {}
        self.interactions = self

    def create(self, **kwargs):
        self.created = kwargs
        return SimpleNamespace(id="interaction-1")

    def get(self, interaction_id):
        assert interaction_id == "interaction-1"
        return SimpleNamespace(
            status="completed",
            outputs=[
                SimpleNamespace(
                    text=(
                        "# Managed Report\n\n"
                        "Managed Gemini research returned a cited report. [1]\n\n"
                        "## Sources\n\n"
                        "[1] Managed Source: https://example.com/source\n"
                    )
                )
            ],
        )


class FakeSemanticJudge:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        return SimpleNamespace(content=json.dumps(self.payload))


class InvalidSemanticJudge:
    def invoke(self, _messages):
        return SimpleNamespace(content='{"cards": [{"id": 1, "keep": true}')


class QuotaSemanticJudge:
    def invoke(self, _messages):
        raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 37s.")


class RaisingSemanticJudge:
    def invoke(self, _messages):
        raise AssertionError("large evidence decks should not invoke the semantic judge")
