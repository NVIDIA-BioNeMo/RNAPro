### Post-processing: Fixing O2' geometry

RNAPro predictions can have incorrect O2′ placement. Run the step below before all-atom LDDT or any analysis that needs correct ribose atoms.

The script removes O2′ and rebuilds it with [Arena](https://github.com/pylelab/Arena). Backbone and base atoms are not moved.

---

### Prerequisites

1. **Python dependencies** (in addition to RNAPro's environment):

   ```bash
   pip install biopython
   ```

2. **Arena** — required for O2' reconstruction. Install and compile:

   ```bash
   git clone https://github.com/pylelab/Arena.git
   cd Arena
   make Arena
   ```

   The script searches for the `Arena` binary in common locations (`~/src/Arena/Arena`, `./Arena/Arena`, etc.). Ensure the compiled binary is on that path, or place/symlink it where the script can find it.

---

### Script: `fix_o2prime.py`

Save the following as `fix_o2prime.py` (or copy from your working directory):

```python
#!/usr/bin/env python3
"""
fix_o2prime.py — Strip O2' atoms from RNA structure files and rebuild them
using Arena.

Bad O2' geometry (e.g. from early Protenix checkpoints) causes all-atom LDDT
to report 0.0; rebuilding from the remaining heavy atoms via Arena restores
meaningful LDDT while leaving backbone / base atoms untouched.

Usage:
    python3 fix_o2prime.py file1.cif file2.pdb ...
    python3 fix_o2prime.py *.cif --outdir /tmp/fixed/

Output files are named {stem}.fixO2prime.{ext} (same format as input).

Note: Arena requires RNA-only input. Use --strip-nonrna if the structure
contains protein or ligand chains. Those chains are removed before Arena and
are NOT restored in the output (the fixed file is suitable for LDDT scoring
but not for visualising the full complex).
"""

import os
import sys
import tempfile
import argparse
import subprocess
from pathlib import Path

from Bio.PDB import MMCIFParser, PDBParser, PDBIO, MMCIFIO


# Standard + common modified RNA residue names
RNA_RESIDUES = {
    'A', 'C', 'G', 'U',
    'PSU', '5MU', 'H2U', 'M2G', 'YG', 'OMG', 'OMC', 'OMU',
    '2MG', '7MG', 'A2M', 'G7M', 'MA6', 'I', 'INO',
}


def look_for_arena():
    candidates = [
        os.path.expanduser("~/src/"),
        os.path.expanduser("~/"),
        "./",
        "../",
        "../../",
        "../../../",
    ]

    for candidate in candidates:
        path = Path(candidate) / "Arena" / "Arena"
        if os.path.exists(path):
            return path

    cmds = [
        "git clone https://github.com/pylelab/Arena.git",
        "cd Arena",
        "make Arena",
    ]
    sys.exit("Arena not found, you can install it using:\n" + "\n".join(cmds))


def strip_o2prime(structure):
    removed = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if "O2'" in residue:
                    residue.detach_child("O2'")
                    removed += 1
    return removed


def strip_nonrna(structure):
    for model in structure:
        empty_chains = []
        for chain in model:
            non_rna = [r.id for r in chain if r.resname.strip() not in RNA_RESIDUES]
            for rid in non_rna:
                chain.detach_child(rid)
            if not list(chain.get_residues()):
                empty_chains.append(chain.id)
        for cid in empty_chains:
            model.detach_child(cid)


def load_structure(filepath: Path):
    ext = filepath.suffix.lower()
    if ext in ('.cif', '.mmcif'):
        return MMCIFParser(QUIET=True).get_structure(filepath.stem, str(filepath)), True
    return PDBParser(QUIET=True).get_structure(filepath.stem, str(filepath)), False


def save_structure(structure, outpath: Path, is_cif: bool):
    if is_cif:
        io = MMCIFIO()
    else:
        io = PDBIO()
    io.set_structure(structure)
    io.save(str(outpath))


def run_arena(arena_path: Path, input_pdb: Path, output_pdb: Path, option: int = 5):
    result = subprocess.run(
        [arena_path, str(input_pdb), str(output_pdb), str(option)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Arena exited {result.returncode}: {result.stderr[:500]}")
    if not output_pdb.exists():
        raise RuntimeError("Arena ran but produced no output file")


def process_file(
    arena_path: Path,
    filepath: Path,
    outdir: Path | None,
    do_strip_nonrna: bool,
    arena_option: int,
    verbose: bool = False,
) -> Path:
    structure, is_cif = load_structure(filepath)

    if do_strip_nonrna:
        strip_nonrna(structure)

    n_removed = strip_o2prime(structure)
    if verbose:
        print(f"  {filepath.name}: stripped {n_removed} O2' atoms")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_in = Path(tmpdir) / "stripped.pdb"
        tmp_out = Path(tmpdir) / "arena_out.pdb"

        # Write RNA-only PDB for Arena
        pdbio = PDBIO()
        pdbio.set_structure(structure)
        pdbio.save(str(tmp_in))

        run_arena(arena_path=arena_path, input_pdb=tmp_in, output_pdb=tmp_out, option=arena_option)

        fixed = PDBParser(QUIET=True).get_structure("fixed", str(tmp_out))

    # Count how many O2' Arena added back
    n_added = sum(1 for m in fixed for ch in m for r in ch if "O2'" in r)
    if verbose:
        print(f"  {filepath.name}: Arena rebuilt {n_added} O2' atoms")

    out_name = filepath.stem + ".fixO2prime" + filepath.suffix
    out_path = (outdir / out_name) if outdir else filepath.parent / out_name
    save_structure(fixed, out_path, is_cif)
    if verbose:
        print(f"  → {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("files", nargs="+", help="CIF or PDB input files")
    ap.add_argument("--outdir", default=None,
                    help="Output directory (default: same directory as each input file)")
    ap.add_argument("--strip-nonrna", action="store_true",
                    help="Remove non-RNA chains before Arena (needed for RNA+protein targets)")
    ap.add_argument("--arena-option", type=int, default=5, choices=[1, 2, 3, 4, 5, 7],
                    help="Arena reconstruction option (default 5)")
    ap.add_argument("--verbose", action="store_true", help="Print verbose output")
    args = ap.parse_args()

    arena_path = look_for_arena()
    if args.verbose:
        print(f"Found Arena in {arena_path}")

    outdir = Path(args.outdir) if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f"Processing {len(args.files)} files in {outdir}")

    errors = 0
    for f in args.files:
        try:
            process_file(arena_path, Path(f).resolve(), outdir, args.strip_nonrna, args.arena_option, args.verbose)
        except Exception as e:
            print(f"  ERROR {f}: {e}", file=sys.stderr)
            errors += 1

    if errors:
        sys.exit(f"{errors} file(s) failed")


if __name__ == "__main__":
    main()
```

---

### Usage

After RNAPro inference, run the script on the output structure(s):

```bash
# Single CIF from inference
python3 fix_o2prime.py path/to/prediction.cif --verbose

# Multiple files, custom output directory
python3 fix_o2prime.py predictions/*.cif --outdir ./fixed/ --verbose

# RNA + protein complex (strip non-RNA for Arena; output is RNA-only)
python3 fix_o2prime.py complex.pdb --strip-nonrna --verbose
```

**Output naming:** `{original_stem}.fixO2prime.{ext}` (same format as input, e.g. `.cif` or `.pdb`).

**Recommended workflow:**

1. Run RNAPro inference → obtain `.cif` / `.pdb` predictions.
2. Run `fix_o2prime.py` on those files (**required** if you evaluate all-atom LDDT or care about ribose geometry).
3. Use the `.fixO2prime.*` files for LDDT and other all-atom metrics.

---

### Options

| Flag | Description |
|------|-------------|
| `--outdir DIR` | Write fixed files to `DIR` instead of next to each input |
| `--strip-nonrna` | Remove protein/ligand chains before Arena (output will not contain them) |
| `--arena-option N` | Arena mode (default `5`; see [Arena README](https://github.com/pylelab/Arena)) |
| `--verbose` | Print strip/rebuild counts per file |

---

### References

- **Arena:** Zion R Perry, Anna Marie Pyle, Chengxin Zhang (2023). *Arena: rapid and accurate reconstruction of full atomic RNA structures from coarse-grained models.* Journal of Molecular Biology. [GitHub](https://github.com/pylelab/Arena) · [Zenodo](https://doi.org/10.5281/zenodo.7566670)
