#!/bin/zsh
# Launch with the project's model-compatible Python environment.
cd -- "${0:A:h}" || exit 1
if [[ ! -x .venv/bin/python ]]; then
    echo 'Set up the app first: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt'
    exit 1
fi
exec .venv/bin/python -m streamlit run bagging.py "$@"
