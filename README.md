# OneFit Connect: Before vs After Dashboard (Team 4)

Static dashboard (HTML + bundled Chart.js, works offline). Compares the baseline 2,700-session dataset with the projected improved dataset.

## Try it locally
Open `index.html` in a browser.

## Deploy free on GitHub Pages (no git commands needed)
1. On github.com click **New repository**, name it `onefit-connect-dashboard`, set **Public**, click **Create repository**.
2. Click **uploading an existing file**.
3. Unzip this package, select EVERYTHING inside the folder (`index.html`, `vendor`, `data`, `scripts`, `README.md`) and drag it into the browser window. Wait for the upload to finish.
4. Click **Commit changes**.
5. **Settings > Pages**: Source **Deploy from a branch**, branch **main**, folder **/ (root)**, **Save**.
6. Your link appears after 1 to 2 minutes: `https://YOUR-USERNAME.github.io/onefit-connect-dashboard/`

## Contents
- `index.html`: the dashboard (overview + 6 chapters + method)
- `vendor/chart.umd.js`: Chart.js (bundled so no internet is needed)
- `data/`: before and after CSVs, one Excel file (before, after, assumptions), `summary.js` used by the page
- `scripts/`: Python that built both datasets. `generate_improved_data.py` lists every assumption at the top.

## Rebuild the data after changing an assumption
```
cd scripts
pip install pandas numpy
python3 generate_improved_data.py
```
This writes new CSVs and `summary.json` in the same folder. Copy the CSVs into `data/` and convert `summary.json` into `data/summary.js` as `window.DATA = <json>;`.

## Important
The "after" numbers are a rule-based projection from stated assumptions, not measured results. The dashboard says so on the Overview and in chapter 7.
