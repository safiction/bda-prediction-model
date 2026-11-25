import unittest

from src.demo import next_action


class DemoScheduleTests(unittest.TestCase):
    def test_order_does_not_skip_training_or_freezing(self):
        options = {action: {"enabled": False} for action in ("train", "prepare", "freeze", "evaluate", "snapshots")}
        options["train"]["ready"] = True
        for action in ("prepare", "train", "freeze", "evaluate"):
            options[action]["enabled"] = True
            self.assertEqual(next_action(options), action)
            options[action]["enabled"] = False
        with self.assertRaises(ValueError):
            next_action(options)
        options["snapshots"]["enabled"] = True
        options["train"]["ready"] = False
        self.assertEqual(next_action(options), "snapshots")
