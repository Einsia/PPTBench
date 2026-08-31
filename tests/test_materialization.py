import hashlib
from io import BytesIO
from pathlib import Path
import tarfile
import tempfile
import unittest

from PIL import Image

from arxiv_flow_dataset.task_materializer import (
    _find_source_pdf,
    pixel_sha256,
    reference_pixel_hashes,
    verify_materialized_task,
)


def archive(members):
    stream = BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as output:
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            output.addfile(info, BytesIO(content))
    return stream.getvalue()


class SourceSelectionTest(unittest.TestCase):
    def test_duplicate_identical_members_are_one_frozen_input(self):
        content = b"%PDF-1.7 frozen figure"
        payload = archive([("a/figure.pdf", content), ("b/figure.pdf", content)])
        self.assertEqual(
            _find_source_pdf(
                payload,
                source_filename="figure.pdf",
                expected_sha256=hashlib.sha256(content).hexdigest(),
            ),
            content,
        )

    def test_wrong_bytes_never_replace_frozen_figure(self):
        with self.assertRaises(ValueError):
            _find_source_pdf(
                archive([("figure.pdf", b"other version")]),
                source_filename="figure.pdf",
                expected_sha256="0" * 64,
            )

    def test_only_frozen_reference_hashes_are_accepted(self):
        hashes = reference_pixel_hashes(
            {
                "pixel_sha256": "renderer",
                "historic_pixel_sha256": "original",
                "other": "unverified",
            }
        )
        self.assertEqual(hashes, {"renderer", "original"})
        self.assertEqual(reference_pixel_hashes({"pixel_sha256": "only"}), {"only"})

    def test_resume_checks_saved_pixels_instead_of_trusting_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "resources").mkdir()
            (root / "source.pdf").write_bytes(b"source")
            original = Image.new("RGB", (4, 4), "white")
            original.save(root / "reference.png")
            (root / "resources/index.json").write_text('{"assets": []}')
            for filename in ("metadata.json", "materialization_report.json"):
                (root / filename).write_text('{"task_id": "task_0004"}')
            spec = {
                "task_id": "task_0004",
                "source": {"sha256": hashlib.sha256(b"source").hexdigest()},
                "reference": {"pixel_sha256": pixel_sha256(original)},
                "resources": [],
            }
            self.assertEqual(verify_materialized_task(root, spec)["task_id"], "task_0004")
            Image.new("RGB", (4, 4), "black").save(root / "reference.png")
            with self.assertRaisesRegex(ValueError, "reference hash mismatch"):
                verify_materialized_task(root, spec)


if __name__ == "__main__":
    unittest.main()
