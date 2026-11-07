import pytest

from efov import parfile


def test_overrides_replace_values_in_place_and_run_lines_follow(tmp_path):
    model = tmp_path / "model.par"
    model.write_text("# header\nbidomain = 2   # pseudo\ndt = 20\nphys_region[0].name = intracellular\n")
    assert parfile.readPar(model) == {"bidomain": "2", "dt": "20", "phys_region[0].name": "intracellular"}
    text = parfile.resolvedText(model, {"bidomain": 0}, {"meshname": "/m/slab", "tsav[0]": 1050.0})
    assert "bidomain = 0  # [override; model.par: 2]" in text and "dt = 20" in text
    assert text.index("meshname = /m/slab") > text.index("dt = 20") and "tsav[0] = 1050.0" in text
    resolved = tmp_path / "run.par"
    resolved.write_text(text)
    assert parfile.readPar(resolved)["bidomain"] == "0"


def test_unknown_override_and_clashing_run_line_are_errors(tmp_path):
    model = tmp_path / "model.par"
    model.write_text("dt = 20\n")
    with pytest.raises(KeyError):
        parfile.resolvedText(model, {"spacedt": 1}, {})
    with pytest.raises(KeyError):
        parfile.resolvedText(model, {}, {"dt": 10})
    model.write_text("dt = 20\ndt = 10\n")
    with pytest.raises(ValueError):
        parfile.readPar(model)


def test_set_value_rewrites_one_key(tmp_path):
    model = tmp_path / "model.par"
    model.write_text("a = 1  # old\nb = 2\n")
    parfile.setValue(model, "a", 0.5, "tuned")
    assert parfile.readPar(model) == {"a": "0.5", "b": "2"}


def test_model_par_files_parse():
    from helpers import ROOT
    main = parfile.readPar(ROOT / "config" / "model.par")
    laplace = parfile.readPar(ROOT / "config" / "leadfield.par")
    assert main["bidomain"] == "0" and main["dataout_e"] == "0" and main["imp_region[0].im"] == "Courtemanche"
    assert laplace["experiment"] == "2" and laplace["stim[1].crct.type"] == "2" and laplace["cg_norm_ellip"] == "2"
    assert not any(k.startswith("gregion") for k in laplace)       # written per run from the model conductivities
