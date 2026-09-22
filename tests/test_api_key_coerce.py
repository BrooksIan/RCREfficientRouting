from rcr_router.cloudera import coerce_api_key, is_plausible_api_key


def test_reject_url_as_api_key(monkeypatch):
    monkeypatch.setenv("CDP_TOKEN", "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.fake.sig")
    monkeypatch.delenv("CLOUDERA_AI_INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert not is_plausible_api_key("https://example.com/endpoints/foo/v1")
    assert not is_plausible_api_key("")
    assert is_plausible_api_key("eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.fake.sig")
    fixed = coerce_api_key("https://bad.example/v1")
    assert fixed and fixed.startswith("eyJ")
