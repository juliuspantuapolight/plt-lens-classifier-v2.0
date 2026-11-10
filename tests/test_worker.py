import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_worker_reports_each_job_and_survives_failures(tmp_path):
    jobs = "\n".join([
        '{"command": "corrections", "argv": ["%s"]}' % (tmp_path / "missing.json").as_posix(),  # argparse error -> 2
        '{"command": "nope", "argv": []}',  # unknown command -> 1
        'not json',  # bad line -> 1
        '{"command": "corrections", "argv": ["--help"]}',  # clean exit -> 0
    ]) + "\n"

    output = subprocess.run([sys.executable, "-m", "src.worker"], input=jobs, capture_output=True, text=True,
                            cwd=REPO, timeout=300).stdout.splitlines()

    assert output[0] == "@@ready"
    assert [line for line in output if line.startswith("@@done")] == ["@@done 2", "@@done 1", "@@done 1", "@@done 0"]
