@echo off
python -m asv machine --yes
python -m asv run -b LinearRegression --append-samples --no-pull --show-stderr
python -m asv publish -o html

