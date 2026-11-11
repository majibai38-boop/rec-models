"""Install the pinned, checksum-verified official Perfetto UI for offline hosting."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

VERSION = 'v59.1'
BUILD = 'v59.1-1e9c61e98'
SHA256 = 'df4433f5d8d98d2c490249870944c819ff85206f61e55e5983bbcc2338c83904'
URL = f'https://github.com/google/perfetto/releases/download/{VERSION}/perfetto-ui.zip'
UI_ROOT = Path(__file__).parent / 'vendor' / 'perfetto'
UI_URL = f'/perfetto/{BUILD}/'


def installed():
    return all((UI_ROOT/BUILD/name).is_file() for name in
               ('index.html', 'frontend_bundle.js', 'engine_bundle.js', 'trace_processor.wasm'))


def install(archive=None):
    if installed():
        return
    UI_ROOT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=UI_ROOT.parent) as temporary:
        root = Path(temporary)
        package = Path(archive) if archive else root/'perfetto.zip'
        if archive is None:
            with urllib.request.urlopen(URL, timeout=120) as response, package.open('wb') as output:
                shutil.copyfileobj(response, output)
        digest = hashlib.sha256()
        with package.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != SHA256:
            raise ValueError('Perfetto archive SHA256 mismatch')
        target = root/'unpacked'
        with zipfile.ZipFile(package) as bundle:
            for entry in bundle.infolist():
                if not (target/entry.filename).resolve().is_relative_to(target.resolve()):
                    raise ValueError('Invalid archive path')
            bundle.extractall(target)
        if UI_ROOT.exists():
            if installed():
                return
            raise ValueError(f'Incomplete installation at {UI_ROOT}; inspect before replacing')
        target.rename(UI_ROOT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', help='Use a previously downloaded official archive offline')
    args = parser.parse_args()
    install(args.archive)
    print(f'Installed Perfetto {VERSION} at {UI_ROOT}')
