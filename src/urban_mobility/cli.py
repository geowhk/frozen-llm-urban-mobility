"""Explicit stages; default workflows never retrain existing models."""

from pathlib import Path
import argparse
import json
import shutil
from .validation import verify_package, compare_outputs


def stage_inputs(package, output, predictions=True):
    package, output = Path(package).resolve(), Path(output).resolve()
    if output == package or output.is_relative_to(package) or package.is_relative_to(output):
        raise ValueError("Output must be separate from the accepted package.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing output: {output}")
    for required in ["inputs", "config.json", "results/oof_predictions_all_models.parquet"]:
        if not (package / required).exists():
            raise FileNotFoundError(package / required)
    output.mkdir(parents=True)
    shutil.copytree(package / "inputs", output / "inputs")
    shutil.copy2(package / "config.json", output / "config.json")
    if predictions:
        shutil.copytree(package / "results", output / "results")
    else:
        (output / "results").mkdir()
    for folder in ["tables", "figure_data", "figures/main", "figures/supplementary"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="Read-only hashes, fixed OOF and metric checks")
    verify.add_argument("--package", type=Path, required=True)
    for command in ["evaluate", "figures", "reproduce", "fit-poly2"]:
        sub = commands.add_parser(command)
        sub.add_argument("--package", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    checks = verify_package(args.package)
    if args.command == "verify":
        print(json.dumps({"checks_passed": len(checks), "read_only": True}))
        return
    root = stage_inputs(args.package, args.output, predictions=args.command != "fit-poly2")
    if args.command == "fit-poly2":
        from .polynomial import fit_benchmark

        fit_benchmark(root)
    if args.command in ["evaluate", "reproduce", "fit-poly2"]:
        from .evaluation import evaluate_predictions

        evaluate_predictions(root)
    if args.command in ["figures", "reproduce"]:
        from .figures import main as render_figures

        render_figures(root)
    # A refit can legitimately differ across numeric-library versions: do not silently certify it.
    compared = compare_outputs(args.package, root) if args.command != "fit-poly2" else []
    summary = {
        "stage": args.command,
        "input_package": str(args.package.resolve()),
        "checks_passed": len(checks),
        "matching_csv_files": compared,
        "trained_poly2": args.command == "fit-poly2",
        "trained_existing_models": False,
    }
    (root / "run_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
