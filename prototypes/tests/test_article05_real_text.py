import unittest

from prototypes.real_experiments.article05_real_text import prepare_corpus


class Article05TextPreparationTests(unittest.TestCase):
    def test_contiguous_splits_are_disjoint_and_share_train_vocabulary(self):
        text = "abcde" * 200
        result = prepare_corpus(text, train_fraction=0.8, validation_fraction=0.1)

        self.assertEqual(result["split_characters"]["train"], 800)
        self.assertEqual(result["split_characters"]["validation"], 100)
        self.assertEqual(result["split_characters"]["test"], 100)
        self.assertEqual(len(result["encoded"]["train"]) + len(result["encoded"]["validation"]) + len(result["encoded"]["test"]), len(text))
        self.assertEqual(result["vocabulary_size"], 5)

    def test_rejects_characters_missing_from_training_vocabulary(self):
        with self.assertRaisesRegex(ValueError, "outside training vocabulary"):
            prepare_corpus("a" * 90 + "xyz", train_fraction=0.8, validation_fraction=0.1)


if __name__ == "__main__":
    unittest.main()
