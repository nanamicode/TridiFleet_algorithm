import importlib.util
import json
from pathlib import Path
import zipfile
import pytest
spec=importlib.util.spec_from_file_location('desktop',Path(__file__).resolve().parents[1]/'scripts/desktop_start.py')
desktop=importlib.util.module_from_spec(spec);spec.loader.exec_module(desktop)

def test_extract_strips_github_root_and_rejects_traversal(tmp_path):
    archive=tmp_path/'a.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('repo-main/main.py','app=1')
        z.writestr('repo-main/pyproject.toml','[project]')
    desktop.extract_release(archive,tmp_path/'good')
    assert (tmp_path/'good/main.py').read_text()=='app=1'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('repo/../../outside','bad')
    with pytest.raises(ValueError):desktop.extract_release(archive,tmp_path/'bad')
    assert not (tmp_path/'outside').exists()

def test_normal_start_uses_last_installed_update(tmp_path):
    release=tmp_path/'versions'/'v3';release.mkdir(parents=True)
    (release/'main.py').write_text('app=1')
    (tmp_path/'current.json').write_text(json.dumps({'path':str(release)}))
    assert desktop.prepare_release(tmp_path,False)==release
