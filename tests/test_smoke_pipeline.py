"""Smoke test for the full pipeline."""

from titanic_automl.cli import run_pipeline


def test_demo_pipeline_runs():
    """Test that demo pipeline runs without errors."""
    exit_code = run_pipeline(demo_mode=True)
    assert exit_code == 0


def test_full_pipeline_runs():
    """Test that full pipeline runs without errors."""
    exit_code = run_pipeline(demo_mode=False)
    assert exit_code == 0


def test_pipeline_produces_output(capsys):
    """Test that pipeline produces expected output."""
    run_pipeline(demo_mode=True)
    captured = capsys.readouterr()
    
    assert "Titanic AutoML Pipeline" in captured.out
    assert "DEMO" in captured.out
    assert "Best model:" in captured.out
    assert "Pipeline completed successfully!" in captured.out
