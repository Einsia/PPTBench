import importlib.util
from pathlib import Path
import unittest
from PIL import Image
from arxiv_flow_dataset.task_materializer import PdfImageBlock, pixel_sha256

spec = importlib.util.spec_from_file_location(
    "manifest_builder",
    Path(__file__).resolve().parents[1] / "scripts/data/build_materialization_manifest.py",
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ManifestOriginTest(unittest.TestCase):
    def test_equal_alpha_does_not_hide_different_rgb_pixels(self):
        target = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
        wrong = Image.new("RGBA", (4, 4), (0, 255, 0, 255))
        blocks = [
            PdfImageBlock(im.mode, im.width, im.height, pixel_sha256(im), im)
            for im in (wrong, target)
        ]
        origin = builder._find_origin(blocks, target)
        self.assertEqual(origin["source_block"]["pixel_sha256"], pixel_sha256(target))
        self.assertNotEqual(origin["source_block"]["pixel_sha256"], pixel_sha256(wrong))


if __name__ == "__main__":
    unittest.main()
