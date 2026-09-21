from __future__ import annotations

import os

from crossword_agent.config import load_local_env


def test_local_env_loads_values_without_overwriting_shell(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "DEMO_ONE=from-file",
                'DEMO_TWO="quoted value"',
                "export DEMO_THREE=exported",
                "not-valid-key=ignored",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEMO_ONE", "from-shell")

    load_local_env(env_file)

    assert os.environ["DEMO_ONE"] == "from-shell"
    assert os.environ["DEMO_TWO"] == "quoted value"
    assert os.environ["DEMO_THREE"] == "exported"
