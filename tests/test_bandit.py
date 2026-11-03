import json

import pytest

from discovery_agent.sandbox.bandit import STRATEGIES, select_strategy, update_strategy


def test_bandit_persists_rewards_and_selects_best_mean(tmp_path, monkeypatch) -> None:
    path = tmp_path / "runs" / "bandit.json"
    monkeypatch.setattr("discovery_agent.sandbox.bandit.random.random", lambda: 0.9)
    monkeypatch.setattr("discovery_agent.sandbox.bandit.random.choice", lambda choices: choices[0])

    update_strategy(path, "minimal_patch", 0.5)
    update_strategy(path, "rewrite", -0.5)

    assert select_strategy(path, epsilon=0) == "minimal_patch"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["version"] == 1
    assert saved["strategies"]["minimal_patch"] == {"count": 1, "total_reward": 0.5}
    assert saved["strategies"]["rewrite"] == {"count": 1, "total_reward": -0.5}
    assert set(saved["strategies"]) == set(STRATEGIES)


def test_bandit_explores_and_validates_inputs(tmp_path, monkeypatch) -> None:
    path = tmp_path / "bandit.json"
    monkeypatch.setattr("discovery_agent.sandbox.bandit.random.random", lambda: 0)
    monkeypatch.setattr("discovery_agent.sandbox.bandit.random.choice", lambda choices: choices[-1])

    assert select_strategy(path, epsilon=0.1) == STRATEGIES[-1]
    with pytest.raises(ValueError, match="epsilon"):
        select_strategy(path, epsilon=1.1)
    with pytest.raises(ValueError, match="Unknown repair strategy"):
        update_strategy(path, "unknown", 1)
    with pytest.raises(ValueError, match="finite"):
        update_strategy(path, "rewrite", float("nan"))


def test_bandit_reports_corrupt_persisted_state(tmp_path) -> None:
    path = tmp_path / "bandit.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="Unable to read bandit state"):
        select_strategy(path)


def test_bandit_converges_on_best_synthetic_repair_strategy(tmp_path, monkeypatch) -> None:
    path = tmp_path / "bandit.json"
    rewards = {
        "minimal_patch": (0.5, 0.5, 0.5),
        "rewrite": (0.0, 0.0, 0.0),
        "simplify": (-0.5, -0.5, -0.5),
    }
    for strategy, outcomes in rewards.items():
        for reward in outcomes:
            update_strategy(path, strategy, reward)

    monkeypatch.setattr("discovery_agent.sandbox.bandit.random.random", lambda: 0.9)
    assert select_strategy(path, epsilon=0.1) == "minimal_patch"
