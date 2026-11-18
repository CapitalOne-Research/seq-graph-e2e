import gzip
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.process_amazon_reviews import process


class AmazonProcessingTest(unittest.TestCase):
    def test_emits_events_and_helpfulness_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            path = raw / "reviews_Clothing_Shoes_and_Jewelry.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for user, helpful in (("u1", [0, 2]), ("u2", [2, 2])):
                    for product in range(5):
                        handle.write(repr({"reviewerID": user, "asin": f"p{product}",
                                           "unixReviewTime": 100 + product, "overall": 5.0,
                                           "reviewText": f"review {user}", "helpful": helpful}) + "\n")
            outputs = process("clothing", raw, root / "processed")
            events = pd.read_csv(outputs["all_events"])
            labels = pd.read_csv(outputs["all_events_label"])
            self.assertEqual(set(events.event_set), {"personal", "relational"})
            self.assertEqual(dict(zip(labels.uid, labels.fraud_label)), {0: 1, 1: 0})
            self.assertIn("reviewText", events.columns)


if __name__ == "__main__":
    unittest.main()
