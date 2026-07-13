from __future__ import annotations

import subprocess
import sys

import pytest

from cstree.alpha.backends import QlibIntegrationUnavailableError


def test_public_qlib_module_import_does_not_import_optional_runtime() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import cstree.alpha.backends.qlib; "
                "assert not any(n == 'qlib' or n.startswith('qlib.') for n in sys.modules)"
            ),
        ],
        check=True,
    )


def test_missing_qlib_runtime_has_actionable_error(monkeypatch) -> None:
    import cstree.alpha.backends.qlib as qlib_backend_module

    def _missing_runtime():
        raise ModuleNotFoundError("No module named 'qlib'", name="qlib")

    monkeypatch.setattr(qlib_backend_module, "_import_runtime_module", _missing_runtime)
    with pytest.raises(QlibIntegrationUnavailableError, match=r"alpha-research\[qlib\]"):
        qlib_backend_module.QlibTrainerBackend()
