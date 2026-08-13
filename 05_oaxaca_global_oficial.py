"""
Descomposición de Oaxaca-Blinder global bajo la especificación oficial.

Estima la descomposición con la misma especificación empleada en el análisis
por bloques sectoriales, de modo que las cifras globales y las sectoriales sean
directamente comparables.

Especificación:
  - Mincer por segmento con efectos fijos de año, Treatment(reference=2005)
  - WLS ponderado por factor de expansión, errores robustos HC3
  - Dotaciones = sum(media_F - media_I) * beta_F sobre capital humano
  - Componente estructural obtenido como residual

Corresponde a la ecuación de la sección metodológica del Capítulo 4.

Entrada  : BASE_FINAL_CORREGIDA_2025.csv
Salida   : reporte en consola

Tesis de licenciatura en Ingeniería Económica y Financiera
Luis Samuel Barria Gutiérrez — Universidad Autónoma del Estado de Hidalgo
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')


RUTA_CSV = r"C:\Users\REGLAMENTOS\Desktop\BASE_FINAL_CORREGIDA_2025.csv"
VAR_DEP  = "ln_ing_hr_real_wins"
VARS_HK  = ['esc_anios', 'exp', 'exp2']
FORMULA  = f"{VAR_DEP} ~ esc_anios + exp + exp2 + C(anio, Treatment(reference=2005))"

df = pd.read_csv(RUTA_CSV, low_memory=False)
df = df[df['esc_anios'] != 99].copy()
df['exp']  = (df['eda'] - df['esc_anios'] - 6).clip(lower=0)
df['exp2'] = df['exp'] ** 2
df = df[df[VAR_DEP].notna()].copy()

def media_pond(data, col):
    return np.average(data[col], weights=data['factor'])

F = df[df['formal'] == 1]
I = df[df['formal'] == 0]

# Modelos por segmento CON años (idéntico a los bloques)
mod_f = smf.wls(FORMULA, data=F, weights=F['factor']).fit(cov_type='HC3')
mod_i = smf.wls(FORMULA, data=I, weights=I['factor']).fit(cov_type='HC3')

# Descomposición con medias ponderadas
y_f    = media_pond(F, VAR_DEP)
y_i    = media_pond(I, VAR_DEP)
brecha = y_f - y_i

x_f = {v: media_pond(F, v) for v in VARS_HK}
x_i = {v: media_pond(I, v) for v in VARS_HK}

expl    = sum((x_f[v] - x_i[v]) * mod_f.params[v] for v in VARS_HK)
no_expl = brecha - expl
pct_e   = expl    / brecha * 100
pct_ne  = no_expl / brecha * 100

print("="*70)
print("  GLOBAL O-B — ESPECIFICACIÓN CONSISTENTE CON LOS BLOQUES")
print("  (con años, ponderado, no_expl = residual)")
print("="*70)
print(f"\n  Brecha total (ln)          : {brecha:.4f}")
print(f"  Componente explicado (K.H.): {expl:.4f}")
print(f"  Componente no explicado    : {no_expl:.4f}")
print(f"\n  Dotaciones (%)             : {pct_e:.1f}%")
print(f"  Estructural (%)            : {pct_ne:.1f}%")
print(f"  Suma (debe = 100.0)        : {pct_e + pct_ne:.1f}%")
print(f"\n  Retorno educ. formal       : {mod_f.params['esc_anios']*100:.2f}%")
print(f"  Retorno educ. informal     : {mod_i.params['esc_anios']*100:.2f}%")
print(f"  Ratio F/I                  : {mod_f.params['esc_anios']/mod_i.params['esc_anios']:.2f}x")
print("")