import pytest

from sae_comp import cuda_check


def test_cuda_preflight_failure_prints_repair_command(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cuda_check.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cuda_check, "_nvidia_smi", lambda: "RTX 4090, 570.00")
    with pytest.raises(SystemExit, match="torch==2.7.1"):
        cuda_check.main()
    output = capsys.readouterr().out
    assert "CUDA preflight:" in output
    assert "RTX 4090" in output


def test_cuda_preflight_success_reports_selected_gpu(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cuda_check.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(cuda_check.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        cuda_check.torch.cuda, "get_device_name", lambda _device: "RTX 4090"
    )
    monkeypatch.setattr(cuda_check, "_nvidia_smi", lambda: "RTX 4090, 570.00")
    cuda_check.main()
    assert "selected_gpu: RTX 4090" in capsys.readouterr().out
