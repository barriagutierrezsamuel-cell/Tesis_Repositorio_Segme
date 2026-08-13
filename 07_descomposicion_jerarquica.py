"""
Descomposición jerárquica de la brecha salarial.

Parte la brecha total en tres componentes que suman el cien por ciento:

  [1] Entre bloques    segregación sectorial de ambos segmentos
                       sum_b (w_F,b - w_I,b) * media_y_F,b
  [2] Dotaciones intra dotaciones de capital humano dentro del bloque
                       sum_b w_I,b * dotaciones_b
  [3] Estructural intra componente estructural dentro del bloque
                       sum_b w_I,b * estructural_b

Convención, consistente con la descomposición global de referencia formal:
el término entre bloques se evalúa a salarios formales por bloque y los
términos intra a participaciones informales. Los componentes por bloque son
los mismos que produce el script anterior.

Esta partición es complementaria de la descomposición estándar y no la
sustituye: ambas se calculan sobre la misma brecha total pero operan a
niveles de agregación distintos, por lo que sus componentes no son
directamente restables entre sí.

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

bloque_map = {
    6:'B1_Comercio', 7:'B1_Comercio',
    9:'B2_Alta_Cal', 10:'B2_Alta_Cal', 12:'B2_Alta_Cal', 13:'B2_Alta_Cal',
    8:'B3_Masivos', 11:'B3_Masivos', 14:'B3_Masivos', 17:'B3_Masivos',
    18:'B3_Masivos', 19:'B3_Masivos',
    15:'B4_Sociales', 16:'B4_Sociales',
}
etiq = {'B1_Comercio':'Comercio', 'B2_Alta_Cal':'Servicios Alta Calificación',
        'B3_Masivos':'Servicios Masivos', 'B4_Sociales':'Servicios Sociales'}
df['bloque'] = df['rubro'].map(bloque_map)
BLOQUES = ['B1_Comercio','B2_Alta_Cal','B3_Masivos','B4_Sociales']

def mp(data, col):
    return np.average(data[col], weights=data['factor'])

F = df[df['formal'] == 1]
I = df[df['formal'] == 0]
W_F = F['factor'].sum()
W_I = I['factor'].sum()

y_F_glob = mp(F, VAR_DEP)
y_I_glob = mp(I, VAR_DEP)
BRECHA   = y_F_glob - y_I_glob

# Por bloque: shares ponderados, medias, y dotaciones/estructural (Tabla 8)
rows = []
for b in BLOQUES:
    db  = df[df['bloque'] == b]
    dbf = db[db['formal'] == 1]
    dbi = db[db['formal'] == 0]

    wF = dbf['factor'].sum() / W_F          # share del formal en bloque b
    wI = dbi['factor'].sum() / W_I          # share del informal en bloque b
    yF = mp(dbf, VAR_DEP)
    yI = mp(dbi, VAR_DEP)
    brecha_b = yF - yI

    # dotaciones/estructural intra-bloque (idéntico a Tabla 8)
    mf = smf.wls(FORMULA, data=dbf, weights=dbf['factor']).fit(cov_type='HC3')
    xF = {v: mp(dbf, v) for v in VARS_HK}
    xI = {v: mp(dbi, v) for v in VARS_HK}
    dot_b = sum((xF[v] - xI[v]) * mf.params[v] for v in VARS_HK)
    est_b = brecha_b - dot_b

    rows.append({'b': b, 'wF': wF, 'wI': wI, 'yF': yF, 'yI': yI,
                 'brecha_b': brecha_b, 'dot_b': dot_b, 'est_b': est_b})

# Componentes agregados
BETWEEN     = sum((r['wF'] - r['wI']) * r['yF'] for r in rows)
WITHIN      = sum(r['wI'] * r['brecha_b'] for r in rows)
WITHIN_dot  = sum(r['wI'] * r['dot_b'] for r in rows)
WITHIN_est  = sum(r['wI'] * r['est_b'] for r in rows)

print("="*74)
print("  DESCOMPOSICIÓN JERÁRQUICA BETWEEN-WITHIN")
print("  CDMX y Estado de México, 2005–2025")
print("="*74)

print("\n  Shares ponderados por bloque (% del total de cada segmento):")
print(f"  {'Bloque':<30} {'w_Formal':>10} {'w_Informal':>12}")
for r in rows:
    print(f"  {etiq[r['b']]:<30} {r['wF']*100:>9.1f}% {r['wI']*100:>11.1f}%")

print(f"\n  {'─'*70}")
print(f"  Brecha total (ln)                 : {BRECHA:.4f}")
print(f"  {'─'*70}")
print(f"  [1] ENTRE bloques (segregación)   : {BETWEEN:.4f}   ({BETWEEN/BRECHA*100:>5.1f}%)")
print(f"  [2] DENTRO – dotaciones           : {WITHIN_dot:.4f}   ({WITHIN_dot/BRECHA*100:>5.1f}%)")
print(f"  [3] DENTRO – estructural          : {WITHIN_est:.4f}   ({WITHIN_est/BRECHA*100:>5.1f}%)")
print(f"  {'─'*70}")
suma = BETWEEN + WITHIN_dot + WITHIN_est
print(f"  SUMA (debe = brecha total)        : {suma:.4f}   ({suma/BRECHA*100:>5.1f}%)")
print(f"  Within total (dot+est)            : {WITHIN:.4f}   ({WITHIN/BRECHA*100:>5.1f}%)")

print(f"""
  ─────────────────────────────────────────────────────────────────────
  LECTURA:
  El O-B global reparte la brecha en 82.7% dotaciones / 17.3% estructural.
  Esta descomposición muestra que de ese componente de 'dotaciones', una
  parte grande ({BETWEEN/BRECHA*100:.1f}%) es en realidad SEGREGACIÓN: los informales
  están confinados a los sub-sectores de baja calificación. Solo {WITHIN_dot/BRECHA*100:.1f}%
  son diferencias de capital humano DENTRO del mismo sub-sector.
  El estructural intra-sectorial ({WITHIN_est/BRECHA*100:.1f}%) es la segmentación en su
  forma más limpia: mismo capital humano, mismo sub-sector, distinta paga.
  ─────────────────────────────────────────────────────────────────────
""")

# Detalle: contribución de cada bloque al within-estructural
print("  Aporte de cada bloque al estructural intra-sectorial:")
print(f"  {'Bloque':<30} {'w_I':>7} {'est_b':>9} {'w_I·est_b':>11} {'% brecha':>10}")
for r in rows:
    contrib = r['wI'] * r['est_b']
    print(f"  {etiq[r['b']]:<30} {r['wI']*100:>6.1f}% {r['est_b']:>9.4f} "
          f"{contrib:>11.4f} {contrib/BRECHA*100:>9.1f}%")
print()