"""
Brecha salarial por tramo educativo, global y por bloque sectorial.

Estima la brecha entre segmentos dentro de cada tramo de la distribución
educativa, controlando por experiencia y efectos fijos de año, y replica el
ejercicio dentro de cada bloque sectorial.

Criterios de reporte:
  - Mínimo de 200 observaciones por celda
  - Amplitud máxima del intervalo de confianza de 30 puntos porcentuales

Entrada  : BASE_FINAL_CORREGIDA_2025.csv
Salida   : BRECHA_por_tramo_educativo.csv, BRECHA_tramo_por_bloque.csv

Tesis de licenciatura en Ingeniería Económica y Financiera
Luis Samuel Barria Gutiérrez — Universidad Autónoma del Estado de Hidalgo
"""

import os
import sys
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════

RUTA   = r"C:\Users\REGLAMENTOS\Desktop\BASE_FINAL_CORREGIDA_2025.csv"
SALIDA = r"C:\Users\REGLAMENTOS\Desktop\FIGURAS_TESIS"

VD = "ln_ing_hr_real_wins"
FORM_TRAMO = f"{VD} ~ C(tramo) + exp + exp2 + C(anio, Treatment(reference=2005))"

MIN_OBS_CELDA = 200     # criterio R1
MAX_AMPLITUD  = 30.0    # criterio R2, en puntos porcentuales

TRAMOS_CORTES = [-0.1, 5.9, 8.9, 11.9, 15.9, 99]
TRAMOS_ETQ = ['Sin primaria completa', 'Primaria', 'Secundaria',
              'Media superior', 'Superior']

bmap = {6: 'B1', 7: 'B1', 9: 'B2', 10: 'B2', 12: 'B2', 13: 'B2',
        8: 'B3', 11: 'B3', 14: 'B3', 17: 'B3', 18: 'B3', 19: 'B3',
        15: 'B4', 16: 'B4'}
etq = {'B1': 'Comercio', 'B2': 'Servicios de Alta Calificación',
       'B3': 'Servicios Masivos', 'B4': 'Servicios Sociales'}
BL = ['B1', 'B2', 'B3', 'B4']


class Tee:
    def __init__(self, ruta):
        self.terminal, self.log = sys.stdout, open(ruta, 'w', encoding='utf-8')

    def write(self, m):
        self.terminal.write(m); self.log.write(m)

    def flush(self):
        self.terminal.flush(); self.log.flush()


os.makedirs(SALIDA, exist_ok=True)
sys.stdout = Tee(os.path.join(SALIDA, 'DIAGNOSTICO_TABLA5.txt'))

print('\n' + '=' * 78)
print('  DIAGNÓSTICO DE LA TABLA 5')
print('=' * 78)
print(f'\n  Criterios de reportabilidad:')
print(f'    R1. N ≥ {MIN_OBS_CELDA} en cada segmento de la celda.')
print(f'    R2. Amplitud del IC 95% < {MAX_AMPLITUD:.0f} puntos porcentuales.')

if not os.path.exists(RUTA):
    print(f'\n  ✗ ERROR: no se encuentra la base en {RUTA}'); sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════
# CARGA
# ═══════════════════════════════════════════════════════════════════════

print('\nCargando base…')
df = pd.read_csv(RUTA, low_memory=False)
if 'esc_anios' not in df.columns and 'anios_esc' in df.columns:
    df = df.rename(columns={'anios_esc': 'esc_anios'})

df = df[df['esc_anios'] != 99].copy()
df['exp']  = (df['eda'] - df['esc_anios'] - 6).clip(lower=0)
df['exp2'] = df['exp'] ** 2
df = df[df[VD].notna()].copy()
df['tramo'] = pd.cut(df['esc_anios'], bins=TRAMOS_CORTES, labels=TRAMOS_ETQ)
df['bloque'] = df['rubro'].map(bmap)
dfb = df[df['bloque'].notna()].copy()


def mp(d, c):
    return np.average(d[c], weights=d['factor'])


def shares_anio(d):
    p = d.groupby('anio')['factor'].sum()
    return (p / p.sum()).to_dict()


def contraste(modelo, exp, shares, tramo):
    idx = {n: i for i, n in enumerate(modelo.params.index)}
    c = np.zeros(len(modelo.params))
    c[idx['Intercept']] = 1.0
    for n, i in idx.items():
        if n.startswith('C(anio'):
            c[i] = float(shares.get(int(n.split('[T.')[1].rstrip(']')), 0.0))
        elif n.startswith('C(tramo'):
            c[i] = 1.0 if n.split('[T.')[1].rstrip(']') == tramo else 0.0
    c[idx['exp']] = exp
    c[idx['exp2']] = exp ** 2
    return c


def brecha(mod_f, mod_i, exp, shares, tramo):
    cf, ci = contraste(mod_f, exp, shares, tramo), contraste(mod_i, exp, shares, tramo)
    g = float(cf @ mod_f.params.values - ci @ mod_i.params.values)
    v = float(cf @ mod_f.cov_params().values @ cf + ci @ mod_i.cov_params().values @ ci)
    return g, np.sqrt(v)


EXP_REF = mp(df, 'exp')
SHARES_G = shares_anio(df)
print(f'   N = {len(df):,}  |  con bloque asignado = {len(dfb):,}')
print(f'   Punto de evaluación: {EXP_REF:.2f} años de experiencia potencial.')

# ═══════════════════════════════════════════════════════════════════════
# ESTIMACIÓN
# ═══════════════════════════════════════════════════════════════════════

print('\nEstimando por bloque y tramo…')
filas = []

# Global
Fg, Ig = df[df['formal'] == 1], df[df['formal'] == 0]
mFg = smf.wls(FORM_TRAMO, data=Fg, weights=Fg['factor']).fit(cov_type='HC3')
mIg = smf.wls(FORM_TRAMO, data=Ig, weights=Ig['factor']).fit(cov_type='HC3')
nFg, nIg = Fg.groupby('tramo').size(), Ig.groupby('tramo').size()
for t in TRAMOS_ETQ:
    nf_, ni_ = int(nFg.get(t, 0)), int(nIg.get(t, 0))
    if nf_ < MIN_OBS_CELDA or ni_ < MIN_OBS_CELDA:
        filas.append({'bloque': 'GLOBAL', 'tramo': t, 'nf': nf_, 'ni': ni_,
                      'g': np.nan, 'se': np.nan})
        continue
    g_, s_ = brecha(mFg, mIg, EXP_REF, SHARES_G, t)
    filas.append({'bloque': 'GLOBAL', 'tramo': t, 'nf': nf_, 'ni': ni_,
                  'g': g_, 'se': s_})

# Por bloque
for b in BL:
    db = dfb[dfb['bloque'] == b]
    dbf, dbi = db[db['formal'] == 1], db[db['formal'] == 0]
    m_f = smf.wls(FORM_TRAMO, data=dbf, weights=dbf['factor']).fit(cov_type='HC3')
    m_i = smf.wls(FORM_TRAMO, data=dbi, weights=dbi['factor']).fit(cov_type='HC3')
    sh = shares_anio(db)
    nfb, nib = dbf.groupby('tramo').size(), dbi.groupby('tramo').size()
    for t in TRAMOS_ETQ:
        nf_, ni_ = int(nfb.get(t, 0)), int(nib.get(t, 0))
        if nf_ < MIN_OBS_CELDA or ni_ < MIN_OBS_CELDA:
            filas.append({'bloque': etq[b], 'tramo': t, 'nf': nf_, 'ni': ni_,
                          'g': np.nan, 'se': np.nan})
            continue
        g_, s_ = brecha(m_f, m_i, EXP_REF, sh, t)
        filas.append({'bloque': etq[b], 'tramo': t, 'nf': nf_, 'ni': ni_,
                      'g': g_, 'se': s_})

T = pd.DataFrame(filas)
T['pct'] = (np.exp(T['g']) - 1) * 100
T['lo']  = (np.exp(T['g'] - 1.96 * T['se']) - 1) * 100
T['hi']  = (np.exp(T['g'] + 1.96 * T['se']) - 1) * 100
T['amplitud'] = T['hi'] - T['lo']
T['R1'] = (T['nf'] >= MIN_OBS_CELDA) & (T['ni'] >= MIN_OBS_CELDA)
T['R2'] = T['amplitud'] < MAX_AMPLITUD
T['reportable'] = T['R1'] & T['R2']

# ═══════════════════════════════════════════════════════════════════════
# SALIDA
# ═══════════════════════════════════════════════════════════════════════

print('\n' + '═' * 78)
print('  TABLA 5 CON INTERVALOS Y TAMAÑOS DE CELDA')
print('═' * 78)
print(f"\n  {'Bloque':<32}{'Tramo':<23}{'N form.':>9}{'N inf.':>9}"
      f"{'Brecha':>10}{'IC 95%':>20}{'Amp.':>8}   Estado")
print('  ' + '-' * 113)
for bl in ['GLOBAL'] + [etq[b] for b in BL]:
    sub = T[T['bloque'] == bl]
    for _, r in sub.iterrows():
        if not r['R1']:
            estado = f'NO ESTIMABLE (celda insuficiente)'
            print(f"  {bl:<32}{r['tramo']:<23}{r['nf']:>9,}{r['ni']:>9,}"
                  f"{'—':>10}{'—':>20}{'—':>8}   {estado}")
            continue
        estado = 'reportable' if r['reportable'] else 'NO REPORTABLE (IC amplio)'
        print(f"  {bl:<32}{r['tramo']:<23}{r['nf']:>9,}{r['ni']:>9,}"
              f"{r['pct']:>9.1f}%   [{r['lo']:>6.1f}, {r['hi']:>6.1f}]"
              f"{r['amplitud']:>8.1f}   {estado}")
    print()

# ─── Veredicto sobre las dos celdas señaladas ──────────────────────────
print('═' * 78)
print('  VEREDICTO SOBRE LAS CELDAS SEÑALADAS')
print('═' * 78)

for bl, tr, motivo in [
        ('Comercio', 'Superior',
         'salto de gran magnitud respecto del tramo previo'),
        ('Servicios de Alta Calificación', 'Primaria',
         'rompe la monotonía del bloque'),
        ('Servicios de Alta Calificación', 'Secundaria',
         'rompe la monotonía del bloque')]:
    r = T[(T['bloque'] == bl) & (T['tramo'] == tr)]
    if r.empty:
        continue
    r = r.iloc[0]
    print(f'\n  {bl} — {tr}')
    print(f'    Motivo de la revisión: {motivo}.')
    if not r['R1']:
        print(f'    N = {r["nf"]:,} formales / {r["ni"]:,} informales.')
        print('    → NO ESTIMABLE. No se reporta.')
    elif not r['R2']:
        print(f'    Brecha {r["pct"]:.1f}%, IC [{r["lo"]:.1f}, {r["hi"]:.1f}], '
              f'amplitud {r["amplitud"]:.1f} pp.')
        print(f'    N = {r["nf"]:,} formales / {r["ni"]:,} informales.')
        print('    → NO REPORTABLE. El intervalo no permite distinguir un')
        print('      hallazgo de ruido muestral. Márcala como no estimable en')
        print('      la tabla del documento y declara la razón al pie.')
    else:
        print(f'    Brecha {r["pct"]:.1f}%, IC [{r["lo"]:.1f}, {r["hi"]:.1f}], '
              f'amplitud {r["amplitud"]:.1f} pp.')
        print(f'    N = {r["nf"]:,} formales / {r["ni"]:,} informales.')
        print('    → REPORTABLE. La estimación es precisa; el valor es un')
        print('      hallazgo, no un artefacto de celda delgada.')

# ─── Monotonía por bloque, solo sobre celdas reportables ───────────────
print('\n' + '═' * 78)
print('  MONOTONÍA Y APERTURA, SOBRE CELDAS REPORTABLES')
print('═' * 78)
print(f"\n  {'Bloque':<32}{'Celdas':>8}{'Monótona':>11}{'Apertura (pp)':>16}")
print('  ' + '-' * 67)
for bl in ['GLOBAL'] + [etq[b] for b in BL]:
    sub = T[(T['bloque'] == bl) & (T['reportable'])].copy()
    sub = sub.set_index('tramo').reindex(TRAMOS_ETQ).dropna(subset=['g'])
    if len(sub) < 2:
        print(f'  {bl:<32}{len(sub):>8}{"—":>11}{"—":>16}')
        continue
    mono = 'sí' if (np.diff(sub['g'].values) > 0).all() else 'NO'
    apert = sub['pct'].iloc[-1] - sub['pct'].iloc[0]
    print(f'  {bl:<32}{len(sub):>8}{mono:>11}{apert:>15.1f}')
print('\n  >> La apertura mide la diferencia entre el tramo reportable más alto')
print('     y el más bajo dentro de cada bloque. Ordena los bloques según cuánto')
print('     se ensancha la brecha a lo largo de la distribución educativa.')

# ─── Contraste con los retornos lineales ───────────────────────────────
print('\n' + '═' * 78)
print('  CORRESPONDENCIA CON LOS RETORNOS A LA ESCOLARIDAD')
print('═' * 78)
FORM_LIN = f"{VD} ~ esc_anios + exp + exp2 + C(anio, Treatment(reference=2005))"
print(f"\n  {'Bloque':<32}{'Ret. F':>9}{'Ret. I':>9}{'Razón':>8}{'Apertura (pp)':>16}")
print('  ' + '-' * 74)
for b in BL:
    db = dfb[dfb['bloque'] == b]
    dbf, dbi = db[db['formal'] == 1], db[db['formal'] == 0]
    rf = smf.wls(FORM_LIN, data=dbf, weights=dbf['factor']).fit(
        cov_type='HC3').params['esc_anios'] * 100
    ri = smf.wls(FORM_LIN, data=dbi, weights=dbi['factor']).fit(
        cov_type='HC3').params['esc_anios'] * 100
    sub = T[(T['bloque'] == etq[b]) & (T['reportable'])].copy()
    sub = sub.set_index('tramo').reindex(TRAMOS_ETQ).dropna(subset=['g'])
    ap = sub['pct'].iloc[-1] - sub['pct'].iloc[0] if len(sub) >= 2 else np.nan
    ap_txt = f'{ap:>15.1f}' if np.isfinite(ap) else f'{"—":>16}'
    print(f'  {etq[b]:<32}{rf:>9.2f}{ri:>9.2f}{rf / ri:>8.2f}{ap_txt}')
print('\n  >> Si el orden de la razón de retornos coincide con el orden de la')
print('     apertura, ambos indicadores describen el mismo mecanismo desde')
print('     ángulos distintos, y esa correspondencia es reportable.')

ruta = os.path.join(SALIDA, 'TABLA5_diagnostico.csv')
T.to_csv(ruta, index=False)
print(f'\n  Tabla completa guardada en:\n  {ruta}')
print('=' * 78 + '\n')

sys.stdout.log.close()
sys.stdout = sys.stdout.terminal