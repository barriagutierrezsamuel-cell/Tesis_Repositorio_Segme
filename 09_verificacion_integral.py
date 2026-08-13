"""
Verificación integral de las cifras del capítulo de resultados.

Regenera desde la base todas las cifras reportadas en el capítulo de
resultados bajo la especificación oficial, para que ningún valor entre al
documento sin haber sido producido por este procedimiento. Corresponde a la
verificación descrita en el Anexo B.

Contenido de la salida:
  - Estadísticas descriptivas ponderadas y sin ponderar por segmento
  - Interceptos y R cuadrada por segmento
  - Medianas ponderadas del ingreso por hora
  - Ecuación de Mincer global
  - Descomposición anual y diferencial de retornos por año

Entrada  : BASE_FINAL_CORREGIDA_2025.csv
Salida   : SALIDA_COMPLETA_cap5.txt

Tesis de licenciatura en Ingeniería Económica y Financiera
Luis Samuel Barria Gutiérrez — Universidad Autónoma del Estado de Hidalgo
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')


RUTA = r"C:\Users\REGLAMENTOS\Desktop\BASE_FINAL_CORREGIDA_2025.csv"
VD   = "ln_ing_hr_real_wins"
HK   = ['esc_anios', 'exp', 'exp2']
FORM = f"{VD} ~ esc_anios + exp + exp2 + C(anio, Treatment(reference=2005))"

df = pd.read_csv(RUTA, low_memory=False)
df = df[df['esc_anios'] != 99].copy()
df['exp']  = (df['eda'] - df['esc_anios'] - 6).clip(lower=0)
df['exp2'] = df['exp'] ** 2
df = df[df[VD].notna()].copy()

F = df[df['formal'] == 1]
I = df[df['formal'] == 0]

def mp(d, c):
    return np.average(d[c], weights=d['factor'])

def mediana_pond(d, c):
    o = d[[c, 'factor']].sort_values(c)
    cum = o['factor'].cumsum()
    corte = o['factor'].sum() / 2
    return o.loc[cum >= corte, c].iloc[0]

print("\n" + "="*76)
print("  VERIFICACIÓN INTEGRAL — CAPÍTULO 5")
print("="*76)
print(f"\n  N total: {len(df):,} | Formales: {len(F):,} | Informales: {len(I):,}")

# ─────────────────────────────────────────────────────────────────────────
print("\n" + "="*76)
print("  COLUMNAS DISPONIBLES EN LA BASE")
print("="*76)
print("  " + " | ".join(sorted(df.columns.tolist())))
print("\n  >> Identifica aquí los nombres de las variables de contrato y")
print("     acceso a servicios de salud para completar las descriptivas.\n")

# ─────────────────────────────────────────────────────────────────────────
print("="*76)
print("  DUDA 1 — DESCRIPTIVAS: SIMPLE vs PONDERADA")
print("="*76)
print(f"\n  {'Variable':<22} {'Segmento':<10} {'Simple':>10} {'Ponderada':>12} {'Δ':>9}")
print(f"  {'-'*66}")

vars_desc = []
for v in ['ing_hr_real_wins', 'ing_hr_real', 'esc_anios', 'exp']:
    if v in df.columns:
        vars_desc.append(v)

for v in vars_desc:
    for d, nom in [(F, 'Formal'), (I, 'Informal')]:
        s = d[v].mean()
        p = mp(d, v)
        print(f"  {v:<22} {nom:<10} {s:>10.2f} {p:>12.2f} {p-s:>9.2f}")
    print()

print("  VALORES EN TU TABLA 2 ACTUAL (para comparar):")
print("    Salario:     Formal $52.84  | Informal $32.24")
print("    Escolaridad: Formal 12.20   | Informal 9.02")
print("    Experiencia: Formal 19.45   | Informal 21.50")
print("\n  >> Si la columna 'Ponderada' NO coincide con estos valores,")
print("     la Tabla 2 se regenera completa.\n")

# Brechas en niveles — las dos versiones
sal = 'ing_hr_real_wins' if 'ing_hr_real_wins' in df.columns else 'ing_hr_real'
sF, sI = mp(F, sal), mp(I, sal)
print(f"  Brecha aritmética (razón de medias): {(sF/sI-1)*100:.1f}%   [tu 5.2 dice 63.9%]")
brecha_ln = mp(F, VD) - mp(I, VD)
print(f"  Brecha geométrica exp(Δln)-1       : {(np.exp(brecha_ln)-1)*100:.1f}%   [tu 5.6 decía 54.7%]")
print(f"  Brecha en ln                       : {brecha_ln:.4f}\n")

# ─────────────────────────────────────────────────────────────────────────
print("="*76)
print("  MEDIANAS PONDERADAS (Sección de distribución salarial)")
print("="*76)
print(f"\n  Mediana salario real/hora — Formal  : ${mediana_pond(F, sal):.2f}   [texto dice ~$37]")
print(f"  Mediana salario real/hora — Informal: ${mediana_pond(I, sal):.2f}   [texto dice ~$25]\n")

# ─────────────────────────────────────────────────────────────────────────
print("="*76)
print("  MINCER GLOBAL (Sección premio a la formalidad)")
print("="*76)
g = smf.wls(f"{VD} ~ formal + esc_anios + exp + exp2 + C(anio, Treatment(reference=2005))",
            data=df, weights=df['factor']).fit(cov_type='HC3')
print(f"\n  Coef. formal      : {g.params['formal']:.4f}  (EE {g.bse['formal']:.4f})")
print(f"  Premio exp(β)-1   : {(np.exp(g.params['formal'])-1)*100:.2f}%")
print(f"  Coef. escolaridad : {g.params['esc_anios']:.4f}  -> {g.params['esc_anios']*100:.2f}% (semielast.)")
print(f"  Coef. experiencia : {g.params['exp']:.4f}")
print(f"  Coef. exp²        : {g.params['exp2']:.6f}")
print(f"  R²                : {g.rsquared:.4f}")
print(f"  N                 : {int(g.nobs):,}\n")

# ─────────────────────────────────────────────────────────────────────────
print("="*76)
print("  DUDA 2 — MINCER POR SEGMENTO (especificación oficial)")
print("="*76)
mf = smf.wls(FORM, data=F, weights=F['factor']).fit(cov_type='HC3')
mi = smf.wls(FORM, data=I, weights=I['factor']).fit(cov_type='HC3')

print(f"\n  {'Parámetro':<26} {'Formal':>14} {'Informal':>14}")
print(f"  {'-'*56}")
print(f"  {'Intercepto':<26} {mf.params['Intercept']:>14.4f} {mi.params['Intercept']:>14.4f}")
print(f"  {'Escolaridad':<26} {mf.params['esc_anios']:>14.4f} {mi.params['esc_anios']:>14.4f}")
print(f"  {'  (EE)':<26} {mf.bse['esc_anios']:>14.4f} {mi.bse['esc_anios']:>14.4f}")
print(f"  {'Experiencia':<26} {mf.params['exp']:>14.4f} {mi.params['exp']:>14.4f}")
print(f"  {'Experiencia²':<26} {mf.params['exp2']:>14.6f} {mi.params['exp2']:>14.6f}")
print(f"  {'R²':<26} {mf.rsquared:>14.4f} {mi.rsquared:>14.4f}")
print(f"  {'N':<26} {int(mf.nobs):>14,} {int(mi.nobs):>14,}")
print(f"\n  Retornos educ.: {mf.params['esc_anios']*100:.2f}% formal / "
      f"{mi.params['esc_anios']*100:.2f}% informal")
print(f"  Ratio (sobre coeficientes crudos): "
      f"{mf.params['esc_anios']/mi.params['esc_anios']:.2f}x")
print(f"\n  TU TEXTO ACTUAL DICE: intercepto 2.1083 / 2.6656 ; R² 0.3737 / 0.0856")
print(f"  >> Si no coinciden, esos párrafos de la Sección 5.4 se reescriben.")
print(f"\n  ⚠️ NOTA: con Treatment(ref=2005) el intercepto es el nivel del AÑO BASE,")
print(f"     no 'el salario base del segmento'. Ajusta la interpretación en el texto.\n")

# ─────────────────────────────────────────────────────────────────────────
print("="*76)
print("  DUDA 4 — LOOP ANUAL: DIFERENCIAL DE RETORNOS Y O-B ESTRUCTURAL")
print("="*76)
F_ANUAL = f"{VD} ~ esc_anios + exp + exp2"
anios = sorted(df['anio'].unique())
res = []
for a in anios:
    da = df[df['anio'] == a]
    daf, dai = da[da['formal'] == 1], da[da['formal'] == 0]
    if len(daf) < 100 or len(dai) < 100:
        continue
    m_f = smf.wls(F_ANUAL, data=daf, weights=daf['factor']).fit(cov_type='HC3')
    m_i = smf.wls(F_ANUAL, data=dai, weights=dai['factor']).fit(cov_type='HC3')
    rf, ri = m_f.params['esc_anios'], m_i.params['esc_anios']
    # O-B del año, medias ponderadas, estructural por residual
    br = mp(daf, VD) - mp(dai, VD)
    dot = sum((mp(daf, v) - mp(dai, v)) * m_f.params[v] for v in HK)
    est = br - dot
    res.append({'anio': a, 'ret_f': rf*100, 'ret_i': ri*100,
                'gap': (rf-ri)*100, 'brecha': br,
                'pct_dot': dot/br*100, 'pct_est': est/br*100})

for v in ['tip_con','pre_asa','remune2c','mh_col']:
    print(f"\n=== {v} ===")
    print(df.groupby('formal')[v].value_counts().unstack(0))

df['con_contrato'] = df['tip_con'].isin([2,3,4]).astype(int)
df['acc_salud']    = (df['pre_asa'] == 1).astype(int)
for v in ['con_contrato','acc_salud']:
    for g,n in [(1,'Formal'),(0,'Informal')]:
        d = df[df['formal']==g]
        print(f"{v} {n}: {np.average(d[v], weights=d['factor'])*100:.1f}%")
print(f"Tasa formalidad ponderada: {np.average(df['formal'], weights=df['factor'])*100:.1f}%")

r = pd.DataFrame(res)
print(f"\n  {'Año':>6} {'Ret.F %':>9} {'Ret.I %':>9} {'Gap pp':>9} "
      f"{'Brecha':>9} {'%Dot':>8} {'%Est':>8}")
print(f"  {'-'*62}")
for _, x in r.iterrows():
    print(f"  {int(x['anio']):>6} {x['ret_f']:>9.2f} {x['ret_i']:>9.2f} "
          f"{x['gap']:>9.2f} {x['brecha']:>9.4f} {x['pct_dot']:>7.1f}% {x['pct_est']:>7.1f}%")

print(f"\n  {'-'*62}")
print(f"  Gap promedio (pp)          : {r['gap'].mean():.2f}   [tu texto dice 6.91]")
print(f"  Gap mínimo                 : {r['gap'].min():.2f}  (año {int(r.loc[r['gap'].idxmin(),'anio'])})")
print(f"  Gap máximo                 : {r['gap'].max():.2f}  (año {int(r.loc[r['gap'].idxmax(),'anio'])})")
print(f"  Años con gap negativo      : {(r['gap']<0).sum()}  <-- debe ser 0 para sostener la narrativa")
print(f"  Estructural anual promedio : {r['pct_est'].mean():.1f}%")
print(f"  Años analizados            : {len(r)}")

# Tendencia — ¿hay convergencia?
from scipy import stats as st
sl, ic, rv, pv, se = st.linregress(r['anio'], r['gap'])
print(f"\n  Tendencia del gap: pendiente {sl:.4f} pp/año (p={pv:.3f})")
if pv > 0.10:
    print("  >> NO significativa: respalda la narrativa de persistencia sin convergencia.")
else:
    print("  >> SIGNIFICATIVA: la narrativa de 'sin tendencia' debe matizarse.")

r.to_csv('VERIFICACION_anual.csv', index=False)
print("\n  Serie anual guardada en VERIFICACION_anual.csv")
print("\n" + "="*76)
print("  FIN. Copia toda esta salida.")
print("="*76 + "\n")