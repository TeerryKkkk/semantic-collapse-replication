import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_providers import ProviderRegistry, MODEL_SPECS, OPENROUTER_BASE_URL, DEEPINFRA_BASE_URL


class FakeMessage:
    content = "ok"


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    choices = [FakeChoice()]


class FakeCompletions:
    def __init__(self, owner): self.owner = owner
    def create(self, **kwargs):
        self.owner.calls.append(kwargs)
        return FakeResponse()


class FakeChat:
    def __init__(self, owner): self.completions = FakeCompletions(owner)


class FakeEmbeddings:
    def __init__(self, owner): self.owner = owner
    def create(self, **kwargs):
        self.owner.embedding_calls.append(kwargs)
        class Item: embedding = [1.0, 0.0]
        class R: data = [Item()]
        return R()


class FakeClient:
    created = []
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.embedding_calls = []
        self.chat = FakeChat(self)
        self.embeddings = FakeEmbeddings(self)
        FakeClient.created.append(self)


class ProviderRoutingTests(unittest.TestCase):
    def setUp(self):
        FakeClient.created = []
        self.env = {
            "OPENAI_API_KEY": "test-openai",
            "OPENROUTER_API_KEY": "test-openrouter",
            "DEEPINFRA_API_KEY": "test-deepinfra",
        }

    def test_routes_external_models_to_expected_openai_compatible_endpoints(self):
        registry = ProviderRegistry(
            selected_model_keys=["deepseekv3", "phi4", "gpt56luna"],
            client_factory=FakeClient,
            environ=self.env,
        )
        messages = [{"role": "user", "content": "test"}]
        registry.generate(MODEL_SPECS["deepseekv3"].model_id, messages, temperature=0.9, max_tokens=200)
        registry.generate(MODEL_SPECS["phi4"].model_id, messages, temperature=0.9, max_tokens=200)
        registry.generate(MODEL_SPECS["gpt56luna"].model_id, messages, temperature=0.9, max_tokens=200)
        base_urls = {client.kwargs.get("base_url") for client in FakeClient.created}
        self.assertIn(OPENROUTER_BASE_URL, base_urls)
        self.assertIn(DEEPINFRA_BASE_URL, base_urls)
        luna_client = next(client for client in FakeClient.created if client.kwargs.get("base_url") is None)
        luna_call = next(call for call in luna_client.calls if call["model"] == MODEL_SPECS["gpt56luna"].model_id)
        self.assertEqual(luna_call["reasoning_effort"], "none")
        self.assertEqual(luna_call["max_completion_tokens"], 200)


if __name__ == "__main__":
    unittest.main()
