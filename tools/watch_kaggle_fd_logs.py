"""Stream Kaggle execution logs to the terminal and a local text file."""
import argparse
import contextlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tmp/kaggle_current_deps'))


class Tee:
    def __init__(self, terminal, file):
        self.terminal, self.file = terminal, file

    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', default='nha4von/mfvlr-genface-fd-weights-20261026-001')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'artifacts/genface_fd_weight_kaggle_20261026/remote_live.log')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    with args.output.open('a', encoding='utf-8') as file:
        terminal = sys.stdout
        with contextlib.redirect_stdout(Tee(terminal, file)):
            print('Following Kaggle live logs:', args.kernel, flush=True)
            api.kernels_logs_cli(args.kernel, follow=True)


if __name__ == '__main__':
    main()
