# Loan default prediction UI

Run from this folder:

```sh
streamlit run bagging.py
```

Install the compatible dependencies first on a new environment:

```sh
python3 -m pip install -r requirements.txt
```

The saved model requires scikit-learn 1.7.2. Restart Streamlit after changing
installed dependencies. The model and prediction preprocessing are unchanged.

Alternatively, double-click `start_app.command` to use the local `.venv`.

## Deploy to Streamlit Community Cloud

Use `bagging.py` as the app file and keep `requirements.txt` and `bagging.pkl`
in the same repository directory. `requirements.txt` pins scikit-learn to 1.7.2
because the saved model was created with that version. The model file is
intentionally included in Git; the training dataset remains excluded.

After pushing changes, deploy the latest commit from the correct branch and
app directory. If an existing deployment still reports scikit-learn 1.6.1,
reboot it or clear its app cache so it rebuilds its environment from
`requirements.txt`.
