#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_splitter.py
-------------------
Create a *prototype* train/val/test split from an image dataset while preserving class balance
and the required proportions (e.g., 70/15/15). You can cap the number of files per class to keep
the prototype small.

Assumptions
-----------
- Input directory is structured as:
    input_root/
      class_a/  img1.jpg, img2.png, ...
      class_b/  img3.jpg, ...
      ...
- Files are images (extensions configurable). Non-files are ignored.

Example
-------
python dataset_splitter.py \
  --input_dir "/path/to/images" \
  --output_dir "/path/to/output_split" \
  --train 0.7 --val 0.15 --test 0.15 \
  --max_per_class 50 \
  --seed 42

After running, you'll have:
  output_split/
    train/
      class_a/ ...
      class_b/ ...
    val/
      class_a/ ...
      class_b/ ...
    test/
      class_a/ ...
      class_b/ ...
  split_manifest.txt   (summary of counts)

Notes
-----
- Uses *copy* by default to avoid altering input. Use --move to move instead.
- Rounding uses largest-remainder method to preserve proportions and totals.
- If a class is tiny, some splits may receive 0 files for that class (mathematically unavoidable).
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import random
import math

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prototype train/val/test splitter with class balance")
    p.add_argument("--input_dir", required=True, help="Path to input root directory with class subfolders")
    p.add_argument("--output_dir", required=True, help="Path to output root directory to create the split")
    p.add_argument("--train", type=float, default=0.7, help="Train fraction (default: 0.7)")
    p.add_argument("--val", type=float, default=0.15, help="Validation fraction (default: 0.15)")
    p.add_argument("--test", type=float, default=0.15, help="Test fraction (default: 0.15)")
    p.add_argument("--max_per_class", type=int, default=None, help="Cap files per class (prototype size). Default: use all")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    p.add_argument("--extensions", type=str, default=None, help="Comma-separated list of extensions to include (e.g. .jpg,.png). Default: common image types")
    p.add_argument("--move", action="store_true", help="Move files instead of copying (default: copy)")
    p.add_argument("--dry_run", action="store_true", help="Only print the planned split; no files copied/moved")
    return p.parse_args()

def list_classes(input_dir: Path) -> List[Path]:
    classes = [p for p in input_dir.iterdir() if p.is_dir()]
    classes.sort(key=lambda x: x.name.lower())
    return classes

def list_images(class_dir: Path, allowed_exts) -> List[Path]:
    imgs = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_exts]
    imgs.sort()
    return imgs

def apportion_counts(total: int, fracs: Tuple[float,float,float]) -> Tuple[int,int,int]:
    """Largest remainder method to split 'total' into parts ~ fracs (train, val, test)."""
    raw = [total * f for f in fracs]
    floors = [math.floor(x) for x in raw]
    remainder = total - sum(floors)
    # distribute remainder to the biggest fractional parts
    fracs_part = [x - math.floor(x) for x in raw]
    order = sorted(range(3), key=lambda i: fracs_part[i], reverse=True)
    for i in range(remainder):
        floors[order[i]] += 1
    return tuple(floors)  # (n_train, n_val, n_test)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def copy_or_move(src: Path, dst: Path, move: bool):
    ensure_dir(dst.parent)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))

def main():
    args = parse_args()
    random.seed(args.seed)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.extensions:
        allowed = {e.strip().lower() if e.strip().startswith('.') else '.' + e.strip().lower()
                   for e in args.extensions.split(',') if e.strip()}
    else:
        allowed = IMG_EXTS

    # Validate fractions
    total_frac = args.train + args.val + args.test
    if not (abs(total_frac - 1.0) < 1e-8):
        print(f"[ERROR] Fractions must sum to 1.0, got {total_frac:.6f}", file=sys.stderr)
        sys.exit(1)
    fracs = (args.train, args.val, args.test)

    if not input_dir.exists():
        print(f"[ERROR] Input dir not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    ensure_dir(output_dir)

    classes = list_classes(input_dir)
    if not classes:
        print(f"[ERROR] No class subfolders found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Prepare output dirs
    for split in ("train", "val", "test"):
        ensure_dir(output_dir / split)

    # Collect and split per class
    totals = {"train": 0, "val": 0, "test": 0}
    class_summary: Dict[str, Dict[str, int]] = {}

    for cdir in classes:
        cname = cdir.name
        imgs = list_images(cdir, allowed)
        if not imgs:
            print(f"[WARN] Class '{cname}' has no matching images; skipping.")
            continue

        if args.max_per_class is not None:
            imgs = imgs[:args.max_per_class]

        # Shuffle deterministically
        random.shuffle(imgs)

        n = len(imgs)
        n_train, n_val, n_test = apportion_counts(n, fracs)

        # Slices
        train_files = imgs[:n_train]
        val_files   = imgs[n_train:n_train+n_val]
        test_files  = imgs[n_train+n_val:n_train+n_val+n_test]

        # Record counts
        class_summary[cname] = {
            "total": n,
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }
        totals["train"] += len(train_files)
        totals["val"]   += len(val_files)
        totals["test"]  += len(test_files)

        # Copy/move files
        if not args.dry_run:
            for f in train_files:
                dst = output_dir / "train" / cname / f.name
                copy_or_move(f, dst, move=args.move)
            for f in val_files:
                dst = output_dir / "val" / cname / f.name
                copy_or_move(f, dst, move=args.move)
            for f in test_files:
                dst = output_dir / "test" / cname / f.name
                copy_or_move(f, dst, move=args.move)

    # Write manifest
    manifest = output_dir / "split_manifest.txt"
    with open(manifest, "w", encoding="utf-8") as mf:
        mf.write(f"Input:  {input_dir}\\n")
        mf.write(f"Output: {output_dir}\\n")
        mf.write(f"Fractions: train={fracs[0]:.4f}, val={fracs[1]:.4f}, test={fracs[2]:.4f}\\n")
        if args.max_per_class is not None:
            mf.write(f"Max per class: {args.max_per_class}\\n")
        mf.write(f"Seed: {args.seed}\\n")
        mf.write("\\nPer-class counts (total/train/val/test):\\n")
        for cname in sorted(class_summary.keys()):
            cs = class_summary[cname]
            mf.write(f"  {cname}: {cs['total']} / {cs['train']} / {cs['val']} / {cs['test']}\\n")
        mf.write("\\nTotals:\\n")
        mf.write(f"  train: {totals['train']}\\n")
        mf.write(f"  val:   {totals['val']}\\n")
        mf.write(f"  test:  {totals['test']}\\n")

    # Console summary
    print("\\n=== SPLIT SUMMARY ===")
    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")
    print(f"Fractions: train={fracs[0]:.2f}, val={fracs[1]:.2f}, test={fracs[2]:.2f}")
    if args.max_per_class is not None:
        print(f"Max per class: {args.max_per_class}")
    print("\\nPer-class counts:")
    for cname in sorted(class_summary.keys()):
        cs = class_summary[cname]
        print(f"  {cname:20s}  total={cs['total']:4d}  train={cs['train']:4d}  val={cs['val']:4d}  test={cs['test']:4d}")
    print("\\nTotals: train={train}  val={val}  test={test}".format(**totals))
    print(f"Manifest written to: {manifest}")
    if args.dry_run:
        print("\\n[DRY RUN] No files were copied or moved. Re-run without --dry_run to materialize the split.")

if __name__ == "__main__":
    main()
