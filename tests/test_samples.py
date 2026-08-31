import json
from pathlib import Path
import tempfile
import unittest

from eval_harness.config import BenchmarkConfig
from eval_harness.samples import _task_specs


class SampleSelectionTest(unittest.TestCase):
    def test_selected_task_does_not_require_other_materialized_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "\n".join(json.dumps({"task_id": t}) for t in ["task_0000", "task_0977"])
            )
            task = root / "tasks/task_0977"
            task.mkdir(parents=True)
            (task / "metadata.json").write_text(
                json.dumps({"task_id": "task_0977", "candidate_id": "frozen-figure"})
            )
            for filename in ("reference.png", "source.pdf"):
                (task / filename).write_bytes(b"present")
            config = BenchmarkConfig(manifest, root / "tasks", root / "runs", 2)
            specs = _task_specs(config, selectors=["task_0977"])
            self.assertEqual([s["task_id"] for s in specs], ["task_0977"])
            with self.assertRaises(FileNotFoundError):
                _task_specs(config)


if __name__ == "__main__":
    unittest.main()
