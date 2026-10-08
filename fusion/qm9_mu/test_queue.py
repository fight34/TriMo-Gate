"""Supervisor contract tests, using temporary runs and fake child processes."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_queue


class QueueTests(unittest.TestCase):
    def options(self, root):
        return argparse.Namespace(data=str(root / "cache"), runs=str(root / "runs"),
                                  output=str(root / "queue"), device="cpu",
                                  concurrency=4, max_hours=36)

    def test_evaluation_barrier_and_all_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.options(root)
            history = []

            class FakeProcess:
                pid = 999999999

                def __init__(self, command, **kwargs):
                    self.mode = command[command.index("--mode") + 1]
                    self.out = Path(command[command.index("--out") + 1])
                    if self.mode == "finalize":
                        self.assert_training_barrier()
                    history.append(self.mode)
                    status = "trained" if self.mode == "train" else "completed"
                    (self.out / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")

                def assert_training_barrier(self):
                    statuses = list((root / "runs").glob("*/seed_*/status.json"))
                    if len(statuses) != 18:
                        raise AssertionError("Evaluation started before all training jobs")
                    if any(json.loads(path.read_text())["status"] not in ("trained", "completed")
                           for path in statuses):
                        raise AssertionError("Evaluation started while training incomplete")

                def poll(self):
                    return 0

                def wait(self, timeout=None):
                    return 0

            with patch.object(run_queue.subprocess, "Popen", FakeProcess), \
                 patch.object(run_queue.subprocess, "run") as aggregate, \
                 patch.object(run_queue.time, "sleep"):
                run_queue.run_queue(args)
                aggregate.assert_called_once()
            self.assertEqual(history, ["train"] * 18 + ["finalize"] * 18)
            state = json.loads((root / "queue" / "queue_status.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "completed")
            self.assertFalse((root / "queue" / "queue.lock").exists())

    def test_failure_does_not_evaluate_or_summarize(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.options(root)
            args.concurrency = 1

            class FailedProcess:
                pid = 999999999

                def __init__(self, command, **kwargs):
                    if command[command.index("--mode") + 1] != "train":
                        raise AssertionError("Should never finalize failed training")

                def poll(self):
                    return 1

                def wait(self, timeout=None):
                    return 1

            with patch.object(run_queue.subprocess, "Popen", FailedProcess), \
                 patch.object(run_queue.subprocess, "run") as aggregate:
                with self.assertRaises(RuntimeError):
                    run_queue.run_queue(args)
                aggregate.assert_not_called()
            state = json.loads((root / "queue" / "queue_status.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "failed")
            self.assertFalse((root / "queue" / "queue.lock").exists())

    def test_resume_only_this_jobs_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.options(root)
            job = run_queue.jobs_for(args)[0]
            directory = Path(job["out"])
            directory.mkdir(parents=True)
            (directory / "last.pt").touch()
            command = run_queue.command_for(args, job, "train")
            self.assertEqual(command[-2:], ["--resume", str(directory / "last.pt")])
            self.assertNotIn("--resume", run_queue.command_for(args, job, "finalize"))

    def test_existing_lock_prevents_duplicate_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.options(root)
            output = Path(args.output)
            output.mkdir()
            (output / "queue.lock").write_text("123", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_queue.run_queue(args)
            self.assertEqual((output / "queue.lock").read_text(), "123")


if __name__ == "__main__":
    unittest.main(verbosity=2)
