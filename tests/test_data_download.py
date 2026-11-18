from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DataDownloadTest(unittest.TestCase):
    def test_dry_run_covers_every_published_dataset_and_reported_amazon_category(self):
        result = subprocess.run(
            [sys.executable, "scripts/download_data.py", "--pres-datasets", "all",
             "--raw-amazon", "all", "--dry-run"],
            cwd=ROOT, text=True, capture_output=True, check=True,
        )
        for dataset in ("amazon-clothing", "amazon-electronics", "brightkite", "github", "gowalla"):
            self.assertIn(f"tasks/{dataset}/**", result.stdout)
            self.assertIn(f"processed/{dataset}_all_events.csv", result.stdout)
        for filename in ("reviews_Movies_and_TV.json.gz", "reviews_Electronics.json.gz",
                         "reviews_Clothing_Shoes_and_Jewelry.json.gz"):
            self.assertIn(filename, result.stdout)

    def test_raw_download_requires_explicit_license_acceptance(self):
        result = subprocess.run(
            [sys.executable, "scripts/download_data.py", "--pres-datasets", "brightkite",
             "--raw-amazon", "movies"], cwd=ROOT, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--accept-amazon-license", result.stderr)


if __name__ == "__main__":
    unittest.main()
