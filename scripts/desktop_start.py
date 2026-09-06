"""Windows desktop bootstrap: update, install, keep state, open browser."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import venv
import webbrowser
import zipfile

URL = 'https://github.com/nanamicode/TridiFleet_algorithm/archive/refs/heads/main.zip'
REPO_ROOT = Path(__file__).resolve().parents[1]


def extract_release(archive: Path, target: Path):
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            path = PurePosixPath(item.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in item.filename:
                raise ValueError('Invalid update archive path')
            if len(path.parts) < 2:
                continue
            destination = target.joinpath(*path.parts[1:])
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with z.open(item) as source, destination.open('wb') as output:
                    shutil.copyfileobj(source, output)
    if not (target / 'main.py').exists() or not (target / 'pyproject.toml').exists():
        raise ValueError('Incomplete update archive')


def prepare_release(base: Path, update: bool):
    marker = base / 'current.json'
    if not update:
        if marker.exists():
            selected = Path(json.loads(marker.read_text())['path']).resolve()
            if selected.is_relative_to((base / 'versions').resolve()) and (selected / 'main.py').is_file():
                return selected
        return REPO_ROOT
    print('Baixando a versao atualizada...', flush=True)
    with tempfile.TemporaryDirectory(prefix='tridifleet-') as temp:
        archive = Path(temp) / 'release.zip'
        request = urllib.request.Request(URL, headers={'User-Agent': 'TridiFleet-Desktop'})
        with urllib.request.urlopen(request, timeout=60) as response, archive.open('wb') as f:
            shutil.copyfileobj(response, f)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()[:20]
        release = base / 'versions' / digest
        if not release.exists():
            release.parent.mkdir(parents=True, exist_ok=True)
            staged = Path(tempfile.mkdtemp(prefix='staging-', dir=release.parent))
            try:
                extract_release(archive, staged)
                staged.rename(release)
            finally:
                if staged.exists():
                    shutil.rmtree(staged)
        return release


def open_when_ready(process):
    for _ in range(120):
        if process.poll() is not None:
            return
        try:
            with urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=1) as r:
                if r.status == 200:
                    webbrowser.open('http://127.0.0.1:8000')
                    return
        except OSError:
            pass
        time.sleep(.5)


def main():
    if sys.version_info < (3, 11):
        raise RuntimeError('Instale Python 3.11 ou superior.')
    # Never stop unrelated processes or overwrite a server currently running.
    with socket.socket() as probe:
        try:
            probe.bind(('127.0.0.1', 8000))
        except OSError as exc:
            raise RuntimeError('Feche a janela do servidor anterior e execute novamente. A porta 8000 esta em uso.') from exc
    base = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'TridiFleetLab'
    base.mkdir(parents=True, exist_ok=True)
    source = prepare_release(base, '--update' in sys.argv)
    data = base / 'data'
    # One-time migration happens automatically, without asking for file handling.
    if not data.exists():
        if (REPO_ROOT / 'data').is_dir():
            shutil.copytree(REPO_ROOT / 'data', data)
        else:
            data.mkdir()
    envfile = base / '.env'
    if not envfile.exists():
        previous = REPO_ROOT / '.env'
        if previous.is_file():
            shutil.copy2(previous, envfile)
        else:
            envfile.write_text('TRIDIFLEET_ADMIN_USER=admin\nTRIDIFLEET_ADMIN_PASSWORD=tridifleet-local\n')
    environment = base / 'runtime'
    python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        print('Preparando o ambiente pela primeira vez...', flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    installed = base / 'installed-source.txt'
    fingerprint = str(source) + hashlib.sha256((source / 'pyproject.toml').read_bytes()).hexdigest()
    if not installed.exists() or installed.read_text() != fingerprint:
        subprocess.run([str(python), '-m', 'pip', 'install', '-e', str(source)], check=True)
        installed.write_text(fingerprint)
    if '--update' in sys.argv:
        marker = base / 'current.json'
        temporary = marker.with_suffix('.tmp')
        temporary.write_text(json.dumps({'path': str(source)}))
        os.replace(temporary, marker)
    env = os.environ.copy()
    env['TRIDIFLEET_AUDIT_DB'] = str(data / 'tridifleet_lab.sqlite3')
    env['TRIDIFLEET_CHECKPOINT'] = str(data / 'lab-checkpoint.json.gz')
    # The desktop lab owns a single local learner; do not inherit another Redis deployment.
    env.pop('REDIS_URL', None)
    print('Abrindo TridiFleet. Pode fechar a aba; mantenha esta janela aberta.', flush=True)
    process = subprocess.Popen([str(python), '-m', 'uvicorn', 'main:app', '--host', '127.0.0.1',
        '--port', '8000', '--env-file', str(envfile)], cwd=source, env=env)
    threading.Thread(target=open_when_ready, args=(process,), daemon=True).start()
    try:
        code = process.wait()
    except KeyboardInterrupt:
        # Ctrl+C is also delivered to uvicorn, which saves before exiting.
        try:
            code = process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.terminate()
            code = process.wait()
    if code:
        raise RuntimeError(f'O servidor encerrou com erro {code}. Confira a mensagem acima.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'\nNao foi possivel iniciar: {exc}', flush=True)
        sys.exit(1)
