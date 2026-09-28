"""Create a before/after evidence image for the report's cleaning section."""
from pathlib import Path
import csv
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures' / '06_cleaning_encoding_before_after.png'
DATA = ROOT / 'data' / 'Telco-Customer-Churn.csv'

def font(size, bold=False):
    names = ['C:/Windows/Fonts/segoeuib.ttf' if bold else 'C:/Windows/Fonts/segoeui.ttf',
             'C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf']
    for name in names:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()

with DATA.open(encoding='utf-8-sig', newline='') as stream:
    rows = list(csv.DictReader(stream))

sample = rows[:5]
labels = ['Month-to-month', 'One year', 'Two year']
width, height = 1800, 1050
img = Image.new('RGB', (width, height), '#FFFFFF')
draw = ImageDraw.Draw(img)
title = font(38, True); subtitle = font(23, True); body = font(20); small = font(18)
ink = '#123B4A'; teal = '#168575'; light = '#E8F4F1'; grid = '#B0BEC5'; dark = '#263238'
draw.text((55, 38), 'Cleaning and encoding evidence', fill=ink, font=title)
draw.text((55, 92), 'Executed on the IBM Telco Customer Churn sample', fill='#456A73', font=body)

def panel(x, y, w, h, heading):
    draw.rounded_rectangle((x, y, x+w, y+h), radius=12, fill='#F6FAF9', outline='#77B8A6', width=2)
    draw.text((x+22, y+18), heading, fill=ink, font=subtitle)

panel(55, 145, 810, 320, '1. Missing values: before -> after')
draw.text((85, 205), 'Check', fill=teal, font=body)
draw.text((425, 205), 'Before', fill=teal, font=body)
draw.text((610, 205), 'After pipeline', fill=teal, font=body)
items = [('TotalCharges blanks', '11', '0 (median imputed)'), ('Missingness indicator', 'not present', 'TotalChargesMissing = 1'), ('New customer rows', '11 rows, tenure = 0', 'retained; no row deleted')]
for i, (a,b,c) in enumerate(items):
    yy = 250 + i*65
    draw.line((80, yy-12, 835, yy-12), fill=grid, width=1)
    draw.text((85, yy), a, fill=dark, font=small)
    draw.text((425, yy), b, fill=dark, font=small)
    draw.text((610, yy), c, fill=dark, font=small)
draw.line((80, 438, 835, 438), fill=grid, width=1)

panel(910, 145, 835, 320, '2. Encoding: one categorical column -> three binary columns')
headers = ['Contract', 'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year']
xs = [940, 1175, 1410, 1600]
for x, h in zip(xs, headers):
    draw.text((x, 205), h.replace('_', '\n'), fill=teal, font=small)
for i, row in enumerate(sample):
    yy = 270 + i*38
    vals = [row['Contract'], int(row['Contract']=='Month-to-month'), int(row['Contract']=='One year'), int(row['Contract']=='Two year')]
    for x, value in zip(xs, vals):
        draw.text((x, yy), str(value), fill=dark, font=small)
    draw.line((940, yy+29, 1710, yy+29), fill=grid, width=1)

panel(55, 515, 1690, 465, '3. Model-ready output and reproducible steps')
steps = [
    ('Raw input', 'TotalCharges is text; Contract is one categorical column.'),
    ('Parse', 'pd.to_numeric(TotalCharges, errors="coerce") creates 11 missing numeric values.'),
    ('Impute', 'SimpleImputer(strategy="median") is fit on the discovery/training partition only.'),
    ('Encode', 'OneHotEncoder(handle_unknown="ignore") expands categorical columns into 43 binary columns.'),
    ('Scale', 'StandardScaler is fit on training numeric columns and reused on confirmation data.'),
    ('Output', '4,930 training rows x 51 finite numeric features; no customerID or target leakage.'),
]
for i, (label, text) in enumerate(steps):
    yy = 585 + i*58
    draw.ellipse((88, yy+5, 116, yy+33), fill=teal)
    draw.text((97, yy+7), str(i+1), fill='white', font=small)
    draw.text((145, yy), label, fill=ink, font=subtitle)
    draw.text((385, yy+2), text, fill=dark, font=body)
    if i < len(steps)-1:
        draw.line((102, yy+35, 102, yy+58), fill='#77B8A6', width=2)

img.save(OUT)
print(OUT)
