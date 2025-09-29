from flask import Flask, render_template
import csv
import pathlib

app = Flask(__name__)

CSV_PATH = pathlib.Path(__file__).parent / "data.csv"

def load_csv(path=CSV_PATH):
    """
    Read CSV and return (headers, rows_as_dicts)
    """
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)
    return headers, rows

@app.route("/")
def index():
    headers, rows = load_csv()
    return render_template("index.html", headers=headers, rows=rows)

if __name__ == "__main__":
    # Runs the app when invoked with: python app.py
    app.run(debug=True, host="0.0.0.0", port=5000)
