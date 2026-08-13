"""
Descomposición de Oaxaca-Blinder global con ponderación consistente.

Estima la descomposición ponderando por el factor de expansión de la ENOE
tanto los coeficientes como las medias de las variables explicativas. Bajo WLS
esta consistencia hace que la identidad se cumpla por construcción, de modo que
dotaciones y componente estructural suman exactamente la brecha observada sin
residuo que absorber.

Entrada  : BASE_FINAL_CORREGIDA_2025.csv
Salida   : reporte en consola

Tesis de licenciatura en Ingeniería Económica y Financiera
Luis Samuel Barria Gutiérrez — Universidad Autónoma del Estado de Hidalgo
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from numpy.linalg import lstsq
import warnings
warnings.filterwarnings('ignore')


ARCHIVO = r"C:\Users\REGLAMENTOS\Desktop\BASE_FINAL_CORREGIDA_2025.csv"

df = pd.read_csv(ARCHIVO)
df = df[df['esc_anios'] != 99].copy()
df['exp']  = (df['eda'] - df['esc_anios'] - 6).clip(lower=0)
df['exp2'] = df['exp'] ** 2
df['ln_ing_hr_real'] = np.log(df['ing_hr_real'].replace(0, np.nan))
df = df[df['ln_ing_hr_real_wins'].notna() & df['ln_ing_hr_real'].notna()].copy()

F = df[df['formal'] == 1]
I = df[df['formal'] == 0]
vars_X = ['esc_anios', 'exp', 'exp2']


def oaxaca_B(F, I, depvar, referencia='formal'):
    """Camino B: medias PONDERADAS por factor. Cierra a 100% sin residuo."""
    Xf_mat = np.column_stack([np.ones(len(F)), F[vars_X].values])
    Xi_mat = np.column_stack([np.ones(len(I)), I[vars_X].values])
    Yf = F[depvar].values
    Yi = I[depvar].values
    fac_f = F['factor'].values
    fac_i = I['factor'].values
    wf = np.sqrt(fac_f)
    wi = np.sqrt(fac_i)

    # Coeficientes por WLS ponderado
    bf = lstsq(Xf_mat * wf[:, None], Yf * wf, rcond=None)[0]
    bi = lstsq(Xi_mat * wi[:, None], Yi * wi, rcond=None)[0]

    # MEDIAS PONDERADAS (el cambio de fondo respecto al Camino A)
    Xf_mean = np.average(Xf_mat, weights=fac_f, axis=0)
    Xi_mean = np.average(Xi_mat, weights=fac_i, axis=0)
    Yf_mean = np.average(Yf, weights=fac_f)
    Yi_mean = np.average(Yi, weights=fac_i)
    brecha  = Yf_mean - Yi_mean

    if referencia == 'formal':
        dot = (Xf_mean - Xi_mean) @ bf
        est = Xi_mean @ (bf - bi)
    else:  # referencia informal
        dot = (Xf_mean - Xi_mean) @ bi
        est = Xf_mean @ (bf - bi)

    return {'brecha': brecha, 'dot': dot, 'est': est,
            'pct_dot': dot/brecha*100, 'pct_est': est/brecha*100,
            'suma_check': (dot+est)/brecha*100,   # debe ser 100.00
            'ret_f': bf[1], 'ret_i': bi[1]}


def mincer_global(data, depvar):
    f = f"{depvar} ~ formal + esc_anios + exp + exp2 + C(anio)"
    return smf.wls(f, data=data, weights=data['factor']).fit(cov_type='HC3')


# ─────────────────────────────────────────────────────────────────────────────
print("="*72)
print("  RESULTADO PRINCIPAL — O-B CAMINO B (winsorizado, ref. formal)")
print("="*72)
p = oaxaca_B(F, I, 'ln_ing_hr_real_wins', referencia='formal')
print(f"\n  Brecha total (ln)        : {p['brecha']:.4f}")
print(f"  Dotaciones (ln)          : {p['dot']:.4f}")
print(f"  Estructural (ln)         : {p['est']:.4f}")
print(f"  Dotaciones (%)           : {p['pct_dot']:.1f}%")
print(f"  Estructural (%)          : {p['pct_est']:.1f}%")
print(f"  Check suma (debe=100.00) : {p['suma_check']:.2f}%")
print("")

# ─────────────────────────────────────────────────────────────────────────────
print("="*72)
print("  EJERCICIO 1 — ROBUSTEZ: SALARIO SIN WINSORIZAR")
print("="*72)
res = {}
for etq, dv in [('Winsorizado', 'ln_ing_hr_real_wins'),
                ('Sin winsor.', 'ln_ing_hr_real')]:
    g  = mincer_global(df, dv)
    ob = oaxaca_B(F, I, dv, referencia='formal')
    res[etq] = {
        'premio': (np.exp(g.params['formal'])-1)*100,
        'coef_f': g.params['formal'], 'retg': g.params['esc_anios']*100,
        'rf': ob['ret_f']*100, 'ri': ob['ret_i']*100,
        'ratio': ob['ret_f']/ob['ret_i'], 'br': ob['brecha'],
        'pd': ob['pct_dot'], 'pe': ob['pct_est'], 'r2': g.rsquared, 'n': g.nobs,
    }
w, s = res['Winsorizado'], res['Sin winsor.']
print(f"\n  {'Estadístico':<32} {'Winsorizado':>13} {'Sin winsor.':>13}")
print(f"  {'-'*60}")
for lbl, k, fmt in [
    ('Premio formalidad (%)','premio','{:.2f}'),('Coef. formal (ln)','coef_f','{:.4f}'),
    ('Retorno educ. global (%)','retg','{:.2f}'),('Retorno educ. formal (%)','rf','{:.2f}'),
    ('Retorno educ. informal (%)','ri','{:.2f}'),('Ratio retornos F/I','ratio','{:.2f}'),
    ('Brecha total (ln)','br','{:.4f}'),('Dotaciones (%)','pd','{:.1f}'),
    ('Estructural (%)','pe','{:.1f}'),('R2 global','r2','{:.4f}'),('N','n','{:,.0f}')]:
    print(f"  {lbl:<32} {fmt.format(w[k]):>13} {fmt.format(s[k]):>13}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  EJERCICIO 2 — REFERENCIA ALTERNATIVA (coef. informales)")
print("="*72)
of = oaxaca_B(F, I, 'ln_ing_hr_real_wins', 'formal')
oi = oaxaca_B(F, I, 'ln_ing_hr_real_wins', 'informal')
print(f"\n  {'Componente':<24} {'Ref. formal':>13} {'Ref. informal':>15}")
print(f"  {'-'*54}")
print(f"  {'Dotaciones (%)':<24} {of['pct_dot']:>12.1f}% {oi['pct_dot']:>14.1f}%")
print(f"  {'Estructural (%)':<24} {of['pct_est']:>12.1f}% {oi['pct_est']:>14.1f}%")
print(f"  {'Check suma':<24} {of['suma_check']:>12.2f}% {oi['suma_check']:>14.2f}%")
print("\n  Ambas referencias cierran a 100.00% de forma limpia, sin absorber")
print("  ningún residuo. Eso es lo que distingue al Camino B.\n")