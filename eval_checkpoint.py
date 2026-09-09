import sys
import argparse
from pathlib import Path

SCORE_RESULTS = True

SRC_DIR = Path.cwd() / "src"
sys.path.insert(0, str(SRC_DIR))

import utils
import build
import evaluate

parser = argparse.ArgumentParser(description="Evaluate a mdlARC checkpoint.")
parser.add_argument(
    "checkpoint_path",
    type=Path,
    help="Path to model checkpoint (.pt file), e.g. runs/tiny.epoch010.pt",
)
parser.add_argument(
    "--run_name",
    type=str,
    default="submission_eval",
    help="Name of the eval run (creates runs/<run_name>/ directory).",
)
parser.add_argument(
    "--max_augments",
    type=int,
    default=80,
    help="Number of test-time augmentation variants.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=100,
)
args = parser.parse_args()

cfg = argparse.Namespace(
    name="submission_run",
    data_path=Path("assets/challenges.json"),

    enable_aug=True,
    max_augments=args.max_augments,
    enable_color_aug=True,
    color_apply_to_test=True,
    enable_dihedral_aug=True,
    dihedral_apply_to_test=True,

    inference_temperature=None,
    inference_top_k=None,
    seed=42,
)

eval_result = evaluate.run_evaluation(
    cfg,
    run_name=args.run_name,
    max_augments=args.max_augments,
    data_path=cfg.data_path,
    checkpoint_path=args.checkpoint_path,
    batch_size=args.batch_size,
    splits=["test"],
    task_ids=None,
)
SUBMISSION_FILE = Path(f"runs/{eval_result[0]}/submission.json")
print("Evaluation complete. submission.json generated.")

if SCORE_RESULTS:
    SOLUTIONS_FILE = Path("assets/solutions.json")
    score = utils.score_arc_submission(SOLUTIONS_FILE, SUBMISSION_FILE)
    print(f"Score: {score}")