from pathlib import Path

import yaml

from alpha_research import promotion_gate


def test_promotion_gate_file_config_resolves_relative_paths_from_config_directory(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "research" / "protocols"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "promotion.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "promotion_gate": {
                    "baseline_run": "runs/baseline",
                    "candidate_run": "runs/candidate",
                    "benchmark_report": "evidence/benchmark.json",
                    "baseline_exposure_screen_report": "evidence/baseline_exposure.json",
                    "candidate_exposure_screen_report": "evidence/candidate_exposure.json",
                    "cpcv": {
                        "baseline_report": "evidence/baseline_cpcv.json",
                        "candidate_report": "evidence/candidate_cpcv.json",
                    },
                    "dsr": {
                        "baseline_report": "evidence/baseline_dsr.json",
                        "candidate_report": "evidence/candidate_dsr.json",
                    },
                    "dynamic_ensemble": {
                        "baseline_report": "evidence/baseline_dynamic.json",
                        "candidate_report": "evidence/candidate_dynamic.json",
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    cfg = promotion_gate.load_promotion_gate_config(config_path)

    assert cfg.baseline_run == (config_dir / "runs/baseline").resolve()
    assert cfg.candidate_run == (config_dir / "runs/candidate").resolve()
    assert cfg.benchmark_report == (config_dir / "evidence/benchmark.json").resolve()
    assert cfg.baseline_exposure_screen_report == (
        config_dir / "evidence/baseline_exposure.json"
    ).resolve()
    assert cfg.candidate_exposure_screen_report == (
        config_dir / "evidence/candidate_exposure.json"
    ).resolve()
    assert cfg.cpcv.baseline_report == (config_dir / "evidence/baseline_cpcv.json").resolve()
    assert cfg.cpcv.candidate_report == (config_dir / "evidence/candidate_cpcv.json").resolve()
    assert cfg.dsr.baseline_report == (config_dir / "evidence/baseline_dsr.json").resolve()
    assert cfg.dsr.candidate_report == (config_dir / "evidence/candidate_dsr.json").resolve()
    assert cfg.dynamic_ensemble.baseline_report == (
        config_dir / "evidence/baseline_dynamic.json"
    ).resolve()
    assert cfg.dynamic_ensemble.candidate_report == (
        config_dir / "evidence/candidate_dynamic.json"
    ).resolve()


def test_promotion_gate_mapping_config_keeps_cwd_relative_path_semantics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    cfg = promotion_gate.load_promotion_gate_config(
        {
            "baseline_run": "runs/baseline",
            "candidate_run": "runs/candidate",
            "benchmark_report": "evidence/benchmark.json",
        }
    )

    assert cfg.baseline_run == (tmp_path / "runs/baseline").resolve()
    assert cfg.candidate_run == (tmp_path / "runs/candidate").resolve()
    assert cfg.benchmark_report == (tmp_path / "evidence/benchmark.json").resolve()
