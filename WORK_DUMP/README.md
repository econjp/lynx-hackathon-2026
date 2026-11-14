# Work dump

Everything the team produced on the way to the final submission: research folders, earlier models
(`src/`, `BOMBASTRAT/`, GOAT), audits, notebooks, the white paper and the old root README. Nothing was deleted.

The files were moved here as they were, so scripts in this folder that build paths from the old repo
layout (for example `research/...` or `src/...` relative to the repo root) need to be run with `WORK_DUMP`
as the root, or have their paths adjusted. The final model in `SUBMISSION/` does not depend on anything here.

Useful entry points:
- `BOMBASTRAT/README.md`: the market-neutral half of the final model, 2-minute walkthrough, grading-criteria table
- `BOMBASTRAT/VALIDATION.md`: placebos, bootstraps, reshuffled histories, asset subsets, lag, costs
- `BOMBASTRAT/COMBINE_GOAT.md`: why 50/50 with GOAT
- `src/goat_model.py`: the GOAT half
- `research/eemil/final_combined/`: audits of both models
