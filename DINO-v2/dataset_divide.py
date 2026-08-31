"""Move 25% of the FAKE and REAL training samples into dataset/train2.0.

The source files are *moved* rather than copied, so successfully selected
samples no longer remain in dataset/train.
"""

from pathlib import Path
import random
import shutil


SOURCE_ROOT = Path(r"D:\NTU_project\hackson\DINO-v2\dataset\train")
DESTINATION_ROOT = Path(r"D:\NTU_project\hackson\DINO-v2\dataset\train2.0")
CLASSES = ("FAKE", "REAL")
SPLIT_RATIO = 0.25
RANDOM_SEED = 42


def move_split(class_name: str, rng: random.Random) -> None:
    source_dir = SOURCE_ROOT / class_name
    destination_dir = DESTINATION_ROOT / class_name

    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source folder does not exist: {source_dir}")

    # Only split the files directly within each class folder.  This prevents
    # accidentally treating nested folders as individual samples.
    samples = [path for path in source_dir.iterdir() if path.is_file()]
    move_count = int(len(samples) * SPLIT_RATIO)
    selected = rng.sample(samples, move_count)

    destination_dir.mkdir(parents=True, exist_ok=True)
    duplicate_names = [path.name for path in selected if (destination_dir / path.name).exists()]
    if duplicate_names:
        raise FileExistsError(
            f"Destination already contains files for {class_name}, for example: "
            f"{duplicate_names[0]}. Rename/remove duplicates before running again."
        )

    for sample in selected:
        shutil.move(str(sample), str(destination_dir / sample.name))

    print(
        f"{class_name}: moved {move_count}/{len(samples)} files "
        f"to {destination_dir}"
    )


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    for class_name in CLASSES:
        move_split(class_name, rng)


if __name__ == "__main__":
    main()
