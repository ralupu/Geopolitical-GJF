# scripts/ — Paper Asset Build Scripts

These scripts assemble paper-ready figures and tables from the outputs of the analysis pipeline and copy them into `Paper_LaTeX/figures/` and `Paper_LaTeX/tables/`.

They are run after the subproject scripts have populated `results/`.

## Planned scripts

| Script | Purpose | Depends on |
|--------|---------|-----------|
| `build_figures.py` | Copies and formats all paper-ready figures into `Paper_LaTeX/figures/` | SP02–SP08 |
| `build_tables.py` | Generates `.tex` table files into `Paper_LaTeX/tables/` | SP06–SP08 |
| `build_paper.py` | Master build: runs all asset scripts then compiles LaTeX | All |

## Conventions

- All scripts are run from the `Paper_GFJ/` root directory.
- Figures are saved as `results/figures/Fig_<N>_<name>.png` (paper-ready, committed to git).
- Tables are saved as `Paper_LaTeX/tables/Table<N>_<name>.tex`.
- Scripts should be idempotent (re-running always produces the same output).
