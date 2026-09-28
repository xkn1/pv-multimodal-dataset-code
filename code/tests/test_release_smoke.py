import json
import unittest
from pathlib import Path

import numpy as np

from pv_dataset.day4 import rectilinear_square


ROOT = Path(__file__).resolve().parents[1]


class ReleaseSmokeTests(unittest.TestCase):
    def test_public_config_contains_no_coordinates(self):
        config = json.loads((ROOT / "config/site_config.example.json").read_text(encoding="utf-8"))
        self.assertIsNone(config["site"]["latitude"])
        self.assertIsNone(config["site"]["longitude"])

    def test_rectilinear_output_is_dense_square(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[..., 1] = 150
        output = rectilinear_square(
            image, center_x=31.5, center_y=31.5, radius_90deg=31,
            output_size=16, corner_zenith_degrees=45, fisheye_model="equidistant"
        )
        self.assertEqual(output.shape, (16, 16, 3))
        self.assertTrue(np.all(output[..., 1] == 150))


if __name__ == "__main__":
    unittest.main()
