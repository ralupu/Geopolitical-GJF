#!/usr/bin/env bash
# ============================================================
# init_git.sh — Run this ONCE on your local machine to
# initialize the Paper_GFJ git repository and make the
# first commit.
#
# Usage (from GeopoliticalRisk/Paper_GFJ/):
#   bash init_git.sh
# ============================================================

set -e

# Remove any stale lock files left by the sandbox
rm -f .git/config.lock 2>/dev/null || true

# (Re-)initialize as a standalone repo
git init -b main

# Configure identity
git config user.email "radulupu.ase@gmail.com"
git config user.name "Radu Lupu"

# Stage everything
git add -A

# Initial commit
git commit -m "Phase 0: initial repository scaffold

- Folder structure: data/, subprojects/01-09, results/, scripts/, Paper_LaTeX/
- ACTION_PLAN.md: full 10-phase action plan with specifications for all phases
- README.md: project overview and pipeline documentation
- .gitignore, requirements.txt
- Subproject READMEs (SP01-SP09)
- Paper_LaTeX/main.tex: manuscript skeleton (section stubs)
- Paper_LaTeX/references.bib: seeded from FI paper bibliography"

echo ""
echo "Done. Repository initialized with initial commit."
echo "To push to a remote, run:"
echo "  git remote add origin <your-remote-url>"
echo "  git push -u origin main"
