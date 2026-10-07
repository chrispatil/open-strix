"""Conversation-history offload must land inside WriteGuard's writable state dir.

Regression test for the follow-up to #135: the agent's CompositeBackend wraps a
WriteGuardBackend running in virtual mode, so ``artifacts_root`` has to be a
virtual path (``/state``), not the absolute host path of the state directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

import open_strix.app as app_mod


class DummyAgent:
    async def ainvoke(self, _: dict[str, Any]) -> dict[str, Any]:
        return {"messages": []}


def _build_agent_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[app_mod.OpenStrixApp, Any]:
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> DummyAgent:
        captured.update(kwargs)
        return DummyAgent()

    monkeypatch.setattr(app_mod, "create_deep_agent", fake_create_deep_agent)
    (tmp_path / "config.yaml").write_text(
        "model: test-model\n"
        "folders:\n"
        "  state: rw\n"
        "  skills: rw\n"
        "  blocks: ro\n"
        "  logs: ro\n",
        encoding="utf-8",
    )
    app = app_mod.OpenStrixApp(tmp_path)
    app._create_agent()
    return app, captured["backend"]


def test_artifacts_root_is_virtual_state_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, backend = _build_agent_backend(tmp_path, monkeypatch)
    assert backend.artifacts_root == f"/{app.layout.state_dir_name}"


def test_conversation_history_offload_lands_in_state_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, backend = _build_agent_backend(tmp_path, monkeypatch)
    middleware = SummarizationMiddleware(
        model=GenericFakeChatModel(messages=iter([])),
        backend=backend,
    )
    monkeypatch.setattr(middleware, "_get_thread_id", lambda: "session_test")

    first = middleware._offload_to_backend(backend, [HumanMessage("hello"), AIMessage("hi")])
    # Second offload exercises the download_files + edit (append) path.
    second = middleware._offload_to_backend(backend, [HumanMessage("again"), AIMessage("still here")])

    assert first == "/state/conversation_history/session_test.md"
    assert second == first
    on_disk = app.layout.state_dir / "conversation_history" / "session_test.md"
    content = on_disk.read_text(encoding="utf-8")
    assert "hello" in content
    assert "still here" in content
    assert content.count("## Summarized at") == 2
