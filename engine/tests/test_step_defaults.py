"""Task 17 performance: fewer default steps for FIBO 2 and Ideogram 4, and saved specs that follow."""

import pytest

from dmlx.catalog import FAMILIES, FIBO2_NOTE, IDEOGRAM4_NOTE
from dmlx.registry import _parse

OLD_FIBO2_NOTE = ("New images only; about 8 minutes at 1024 px; follows detailed descriptions and draws text well; "
                  "non-commercial license.")
OLD_IDEOGRAM4_NOTE = ("New images only; about 8.5 minutes at 1024 px; the strongest at typography and exact placement; "
                      "non-commercial license.")


def _saved(adapter: str, steps: int, note: str) -> dict:
    """A spec as the app saved it when the model was added (a snapshot of the catalog then)."""
    return {"id": "m", "name": "M", "adapter": adapter, "repo": "r", "license": "other", "tasks": ["t2i"],
            "components": {"transformer": {"phase": "denoise"}}, "defaults": {"steps": steps, "guidance": 5.0},
            "agent_note": note}


def test_catalog_defaults_are_the_measured_ones():
    assert FAMILIES["fibo2"]["variants"][0]["defaults"]["steps"] == 20
    assert FAMILIES["ideogram4"]["variants"][0]["defaults"]["steps"] == 12
    assert FAMILIES["fibo2"]["agent_note"] == FIBO2_NOTE
    assert FAMILIES["ideogram4"]["agent_note"] == IDEOGRAM4_NOTE


@pytest.mark.parametrize("adapter, old, new, old_note, new_note", [
    ("mflux.fibo2", 30, 20, OLD_FIBO2_NOTE, FIBO2_NOTE),
    ("mflux.ideogram4", 20, 12, OLD_IDEOGRAM4_NOTE, IDEOGRAM4_NOTE),
])
def test_saved_specs_move_to_the_new_steps_but_a_users_choice_stays(adapter, old, new, old_note, new_note):
    spec = _parse(_saved(adapter, old, old_note))
    assert spec.defaults["steps"] == new and spec.agent_note == new_note
    assert _parse(_saved(adapter, 48, old_note)).defaults["steps"] == 48  # set by the user: kept
    assert _parse(_saved(adapter, old, "my own note")).agent_note == "my own note"
    assert _parse(_saved("mflux.flux2_klein", old, old_note)).defaults["steps"] == old  # other families unchanged


def test_ideogram_12_steps_picks_its_turbo_preset():
    from dmlx.adapters.mflux_family import FAMILIES as ADAPTER_FAMILIES

    presets = ADAPTER_FAMILIES["mflux.ideogram4"]["presets"]
    assert presets[min(presets, key=lambda n: abs(n - 12))] == "V4_TURBO_12"
