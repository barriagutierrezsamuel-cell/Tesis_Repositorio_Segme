"""
Robustez: comparación entre la muestra completa y la restringida a asalariados.

Contrasta las cifras principales estimadas sobre el universo de trabajadores
subordinados y remunerados frente a las obtenidas al restringir la muestra a
quienes perciben un salario propiamente dicho, para documentar que la decisión
sobre la población objetivo no determina los resultados.

Entrada  : BASE_FINAL_CORREGIDA_2025.csv
Salida   : reporte en consola

Tesis de licenciatura en Ingeniería Económica y Financiera
Luis Samuel Barria Gutiérrez — Universidad Autónoma del Estado de Hidalgo
"""

import pandas as pd, numpy as np, statsmodels.formula.api as smf
from scipy import stats as st

RUTA = r"C:\Users\REGLAMENTOS\Desktop\BASE_FINAL_CORREGIDA_2025.csv"
VD, HK = "ln_ing_hr_real_wins", ['esc_anios', 'exp', 'exp2']
FORM = f"{VD} ~ esc_anios + exp + exp2 + C(anio, Treatment(reference=2005))"

d = pd.read_csv(RUTA, low_memory=False)
d = d[d['esc_anios'] != 99].copy()
d['exp'] = (d['eda'] - d['esc_anios'] - 6).clip(lower=0)
d['exp2'] = d['exp'] ** 2
d = d[d[VD].notna()].copy()

mp = lambda x, c: np.average(x[c], weights=x['factor'])

print(f"{'Muestra':<34}{'N':>10}{'Brecha':>9}{'Ret.F':>8}{'Ret.I':>8}"
      f"{'Razón':>7}{'%Dot':>8}{'%Est':>8}")
for nom, sub in [('Completa (subordinados y remun.)', d),
                 ('Solo asalariados (remune2c=1)', d[d['remune2c'] == 1])]:
    F, I = sub[sub['formal'] == 1], sub[sub['formal'] == 0]
    mF = smf.wls(FORM, data=F, weights=F['factor']).fit(cov_type='HC3')
    mI = smf.wls(FORM, data=I, weights=I['factor']).fit(cov_type='HC3')
    br = mp(F, VD) - mp(I, VD)
    dot = sum((mp(F, v) - mp(I, v)) * mF.params[v] for v in HK)
    rf, ri = mF.params['esc_anios'] * 100, mI.params['esc_anios'] * 100
    print(f'{nom:<34}{len(sub):>10,}{br:>9.4f}{rf:>8.2f}{ri:>8.2f}'
          f'{rf / ri:>7.2f}{dot / br * 100:>7.1f}%{(br - dot) / br * 100:>7.1f}%')