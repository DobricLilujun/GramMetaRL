from __future__ import annotations

from grammetarl.agent_env import GrammarFirstEnv


def test_env_reset_and_done_flag() -> None:
    env = GrammarFirstEnv(step_budget=2)
    s = env.reset(sample_id="x", src_sentence="I did not see him.")
    assert s["done"] is False


def test_env_return_translation_action() -> None:
    env = GrammarFirstEnv(step_budget=3)
    env.reset(sample_id="x", src_sentence="I did not see him.")
    s, done = env.step("RETURN_TRANSLATION", {"translation": "Ech hunn hien net gesinn."})
    assert done is True
    assert s["final_translation"] == "Ech hunn hien net gesinn."
