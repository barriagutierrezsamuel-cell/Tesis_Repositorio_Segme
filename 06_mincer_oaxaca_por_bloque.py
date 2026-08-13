"""
Estimación de Mincer y descomposición de Oaxaca-Blinder por bloque sectorial.

Estima retornos al capital humano y descompone la brecha salarial dentro de
cada uno de los cuatro bloques de actividad construidos a partir del SCIAN,
además de la estimación global comparable.

Especificación:
  - WLS ponderado por factor de expansión, errores robustos HC3
  - Efectos fijos de año, Treatment(reference=2005)
  - Participaciones sectoriales ponderadas por factor de expansión
  - Validaciones de integridad al inicio de la ejecución

Entrada  : BASE_FINAL_CORREGIDA_2025.csv
Salida   : reporte en consola y figuras por bloque

Tesis de licenciatura en Ingeniería Económica y Financiera
Luis Samuel Barria Gutiérrez — Universidad Autónoma del Estado de Hidalgo
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


RUTA_CSV = r"C:\Users\REGLAMENTOS\Desktop\BASE_FINAL_CORREGIDA_2025.csv"
VAR_DEP  = "ln_ing_hr_real_wins"

# =============================================================================
# 1. CARGA, FILTROS Y VALIDACIÓN ESTRICTA
# =============================================================================
print("\n" + "="*78)
print("  ANÁLISIS POR BLOQUES SCIAN v5 — VERSIÓN FINAL")
print("  CDMX y Estado de México, 2005–2025")
print("="*78 + "\n")

df = pd.read_csv(RUTA_CSV, low_memory=False)
df = df[df['esc_anios'] != 99].copy()
df['exp']  = (df['eda'] - df['esc_anios'] - 6).clip(lower=0)
df['exp2'] = df['exp'] ** 2
df = df[df[VAR_DEP].notna()].copy()

# [6] VALIDACIONES ESTRICTAS
N_ESPERADO   = 245_508
N_FORMAL     = 113_175
N_INFORMAL   = 132_333

try:
    assert len(df) == N_ESPERADO, (
        f"Inconsistencia: se esperaban {N_ESPERADO:,} obs, hay {len(df):,}"
    )
    assert df['formal'].sum() == N_FORMAL, (
        f"Formales: esperados {N_FORMAL:,}, hay {df['formal'].sum():,}"
    )
    assert (df['formal'] == 0).sum() == N_INFORMAL, (
        f"Informales: esperados {N_INFORMAL:,}, hay {(df['formal']==0).sum():,}"
    )
    print(f"✅ Validación: {len(df):,} obs | "
          f"{df['formal'].sum():,} formales | "
          f"{(df['formal']==0).sum():,} informales\n")
except AssertionError as e:
    print(f"❌ ERROR DE VALIDACIÓN: {e}")
    print("   El archivo CSV cambió. Verifica antes de continuar.")
    raise

# =============================================================================
# 2. VARIABLE DE BLOQUE SCIAN
# =============================================================================
bloque_map = {
    6:  'B1_Comercio',    7:  'B1_Comercio',
    9:  'B2_Alta_Cal',    10: 'B2_Alta_Cal',
    12: 'B2_Alta_Cal',    13: 'B2_Alta_Cal',
    8:  'B3_Masivos',     11: 'B3_Masivos',
    14: 'B3_Masivos',     17: 'B3_Masivos',
    18: 'B3_Masivos',     19: 'B3_Masivos',
    15: 'B4_Sociales',    16: 'B4_Sociales',
}
etiq_full = {
    'B1_Comercio': 'Comercio',
    'B2_Alta_Cal': 'Servicios Alta Calificación',
    'B3_Masivos':  'Servicios Masivos',
    'B4_Sociales': 'Servicios Sociales',
}
etiq_corto = {
    'B1_Comercio': 'Comercio',
    'B2_Alta_Cal': 'Alta\nCalificación',
    'B3_Masivos':  'Servicios\nMasivos',
    'B4_Sociales': 'Servicios\nSociales',
}

df['bloque'] = df['rubro'].map(bloque_map)
assert df['bloque'].isna().sum() == 0, (
    "Hay rubros fuera del rango 6-19 que no se mapearon"
)

BLOQUES = ['B1_Comercio', 'B2_Alta_Cal', 'B3_Masivos', 'B4_Sociales']

# [5] AÑO BASE EXPLÍCITO — 2005 como referencia
FORMULA = f"{VAR_DEP} ~ esc_anios + exp + exp2 + C(anio, Treatment(reference=2005))"
VARS_HK = ['esc_anios', 'exp', 'exp2']

def media_pond(data, col):
    """Media ponderada por factor de expansión ENOE."""
    return np.average(data[col], weights=data['factor'])

def prop_pond(data, col):
    """Proporción ponderada por factor de expansión ENOE."""
    return np.average(data[col], weights=data['factor']) * 100

# =============================================================================
# 3. TABLA 6 — DISTRIBUCIÓN PONDERADA POR BLOQUE
#    [1] CORRECCIÓN: % Muestra y % Formal ahora ponderados por factor ENOE
# =============================================================================
print("─"*78)
print("TABLA 6. CARACTERÍSTICAS POR BLOQUE SCIAN (datos ponderados)")
print("─"*78)
print(f"{'Bloque':<32} {'N':>8} {'% Muestra':>10} {'% Formal':>10} "
      f"{'Esc. Formal':>12} {'Esc. Informal':>14}")
print("─"*78)

# Suma total de factores para % Muestra ponderado
total_factor = df['factor'].sum()

for bloque in BLOQUES:
    sub   = df[df['bloque'] == bloque]
    sub_f = sub[sub['formal'] == 1]
    sub_i = sub[sub['formal'] == 0]

    # [1] Ponderación correcta
    pct_muestra = sub['factor'].sum() / total_factor * 100
    pct_formal  = prop_pond(sub, 'formal')

    esc_f = media_pond(sub_f, 'esc_anios') if len(sub_f) > 0 else 0
    esc_i = media_pond(sub_i, 'esc_anios') if len(sub_i) > 0 else 0

    print(f"{etiq_full[bloque]:<32} {len(sub):>8,} "
          f"{pct_muestra:>9.1f}% {pct_formal:>9.1f}% "
          f"{esc_f:>12.2f} {esc_i:>14.2f}")

print("\nNota: % Muestra y % Formal calculados con ponderación por factor de")
print("expansión ENOE (fac_tri). Representativos de la población.\n")

# =============================================================================
# 4. TABLA 7 — MINCER CON WALD TESTS
#    [5] Fórmula con año base 2005 explícito
# =============================================================================
print("─"*78)
print("TABLA 7. RETORNOS AL CAPITAL HUMANO POR BLOQUE — CON PRUEBA WALD")
print("Variable dependiente: ln(salario real por hora, winsorizado)")
print("Método: WLS, errores HC3, efectos fijos de año (ref: 2005).")
print("─"*78)
print(f"{'Bloque':<30} {'Var':<6} {'β_F':>8} {'(SE)':>8} "
      f"{'β_I':>8} {'(SE)':>8} {'Wald':>8} {'p-val':>8} {'Sig':>4}")
print("─"*78)

modelos    = {}
wald_tests = {}

for bloque in BLOQUES:
    df_b = df[df['bloque'] == bloque].copy()
    modelos[bloque] = {}

    for fval, seg in [(1, 'Formal'), (0, 'Informal')]:
        df_seg = df_b[df_b['formal'] == fval].copy()
        if len(df_seg) < 200:
            continue
        res = smf.wls(FORMULA, data=df_seg,
                      weights=df_seg['factor']).fit(cov_type='HC3')
        modelos[bloque][seg] = res

    if 'Formal' not in modelos[bloque] or 'Informal' not in modelos[bloque]:
        continue

    res_f = modelos[bloque]['Formal']
    res_i = modelos[bloque]['Informal']
    wald_tests[bloque] = {}

    first = True
    for var, label in [('esc_anios', 'esc'), ('exp', 'exp')]:
        diff  = res_f.params[var] - res_i.params[var]
        se_d  = np.sqrt(res_f.bse[var]**2 + res_i.bse[var]**2)
        wald  = (diff / se_d)**2
        pval  = 1 - stats.chi2.cdf(wald, df=1)
        sig   = ('***' if pval < 0.01 else '**' if pval < 0.05
                 else '*' if pval < 0.10 else 'ns')
        wald_tests[bloque][var] = {
            'wald': wald, 'pvalue': pval, 'diff': diff
        }

        name = etiq_full[bloque] if first else ''
        print(f"{name:<30} {label:<6} "
              f"{res_f.params[var]:>8.4f} ({res_f.bse[var]:.4f}) "
              f"{res_i.params[var]:>8.4f} ({res_i.bse[var]:.4f}) "
              f"{wald:>8.2f} {pval:>8.4f} {sig:>4}")
        first = False

    print(f"  {'R² Formal':>20}: {modelos[bloque]['Formal'].rsquared:.3f}  "
          f"N={int(modelos[bloque]['Formal'].nobs):,}  |  "
          f"R² Informal: {modelos[bloque]['Informal'].rsquared:.3f}  "
          f"N={int(modelos[bloque]['Informal'].nobs):,}")
    print()

print("Significancia: *** p<0.01 | ** p<0.05 | * p<0.10 | ns = no significativo\n")

# =============================================================================
# 5. TABLA 8 — OAXACA-BLINDER OPCIÓN B
#    [2] CORRECCIÓN: nomenclatura precisa "dotaciones de capital humano"
# =============================================================================
print("─"*78)
print("TABLA 8. DESCOMPOSICIÓN OAXACA-BLINDER POR BLOQUE SCIAN")
print("Twofold con coeficientes formales como referencia.")
print("Comp. explicado    = Σ(X̄_F − X̄_I)·β_F sobre capital humano (esc, exp, exp²)")
print("Comp. no explicado = Brecha − Comp.explicado [residual]")
print("─"*78)
print(f"{'Bloque':<30} {'Brecha ln':>10} {'Expl. K.H.':>11} {'%':>7} "
      f"{'No Expl.':>10} {'%':>7}")
print("─"*78)

ob_res = {}

for bloque in BLOQUES:
    if 'Formal' not in modelos[bloque] or 'Informal' not in modelos[bloque]:
        continue
    df_b = df[df['bloque'] == bloque]
    df_f = df_b[df_b['formal'] == 1]
    df_i = df_b[df_b['formal'] == 0]

    y_f    = media_pond(df_f, VAR_DEP)
    y_i    = media_pond(df_i, VAR_DEP)
    brecha = y_f - y_i

    res_f  = modelos[bloque]['Formal']
    x_f    = {v: media_pond(df_f, v) for v in VARS_HK}
    x_i    = {v: media_pond(df_i, v) for v in VARS_HK}

    expl    = sum((x_f[v] - x_i[v]) * res_f.params[v] for v in VARS_HK)
    no_expl = brecha - expl
    pct_e   = expl    / brecha * 100
    pct_ne  = no_expl / brecha * 100

    ob_res[bloque] = {
        'brecha': brecha, 'expl': expl, 'no_expl': no_expl,
        'pct_e': pct_e,   'pct_ne': pct_ne,
    }

    print(f"{etiq_full[bloque]:<30} {brecha:>10.4f} "
          f"{expl:>11.4f} {pct_e:>6.1f}% "
          f"{no_expl:>10.4f} {pct_ne:>6.1f}%")

# =============================================================================
# Referencia GLOBAL para Figura 11
# [3][7] Global con MISMA especificación que los bloques: con años, ponderado,
#        no_expl = residual. Estructural = un solo número limpio (sin el
#        antiguo desglose 17.4% + 1.7%, que era artefacto de ponderación).
# =============================================================================
GLOBAL_BRECHA      = 0.4297                # media ponderada por factor
GLOBAL_EXPL        = 82.7                  # dotaciones (con años, ponderado)
GLOBAL_NEXPL_AGG   = 100.0 - GLOBAL_EXPL   # 17.3% estructural

print(f"\n  {'GLOBAL (modelo principal)':<28} "
      f"{GLOBAL_BRECHA:>10.4f} "
      f"{'0.3554':>11} {GLOBAL_EXPL:>6.1f}% "
      f"{'0.0743':>10} {GLOBAL_NEXPL_AGG:>6.1f}%")
print(f"\n  Nota: El componente no explicado de {GLOBAL_NEXPL_AGG:.1f}% refleja")
print(f"        diferencias en la remuneración del capital humano y en el")
print(f"        intercepto entre segmentos. Ver Tabla 5 (Capítulo 5).\n")

# =============================================================================
# 6. FIGURA 10 — RETORNOS POR BLOQUE
#    [4] CORRECCIÓN: n.s. en gris neutro, no en rojo
# =============================================================================
print("🎨 Generando Figura 10...")

ret_f  = [modelos[b]['Formal'].params['esc_anios'] * 100 for b in BLOQUES]
ret_i  = [modelos[b]['Informal'].params['esc_anios'] * 100 for b in BLOQUES]
se_f   = [modelos[b]['Formal'].bse['esc_anios'] * 100 for b in BLOQUES]
se_i   = [modelos[b]['Informal'].bse['esc_anios'] * 100 for b in BLOQUES]
ratios = [f/i for f, i in zip(ret_f, ret_i)]
pvals  = [wald_tests[b]['esc_anios']['pvalue'] for b in BLOQUES]
etiq_x = [etiq_corto[b] for b in BLOQUES]

COLOR_F    = '#1a2e4a'
COLOR_I    = '#8fa8c8'
COLOR_SIG  = '#c0392b'
COLOR_NS   = '#666666'   # [4] Gris neutro para n.s.

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(11, 9),
    gridspec_kw={'height_ratios': [2.2, 1]},
    constrained_layout=True
)

x = np.arange(len(BLOQUES))
w = 0.36

# ── Panel superior — barras con IC 95% ──
b_f = ax1.bar(x - w/2, ret_f, w,
              yerr=[s * 1.96 for s in se_f], capsize=4,
              color=COLOR_F, label='Formal',
              error_kw={'ecolor': '#333', 'lw': 1.2}, zorder=3)
b_i = ax1.bar(x + w/2, ret_i, w,
              yerr=[s * 1.96 for s in se_i], capsize=4,
              color=COLOR_I, label='Informal',
              error_kw={'ecolor': '#666', 'lw': 1.2}, zorder=3)

for bar, val in zip(b_f, ret_f):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + max(se_f)*1.96 + 0.3,
             f'{val:.2f}%', ha='center', va='bottom',
             fontsize=9.5, fontweight='bold', color=COLOR_F)
for bar, val in zip(b_i, ret_i):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + max(se_i)*1.96 + 0.3,
             f'{val:.2f}%', ha='center', va='bottom',
             fontsize=9.5, fontweight='bold', color='#4a6a8a')

# [4] Pruebas de significancia — n.s. en gris, ¬ rojo
for xi, p in zip(x, pvals):
    sig   = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else 'n.s.'
    color = '#000' if p < 0.10 else COLOR_NS
    ax1.text(xi, -0.7, sig, ha='center', va='top',
             fontsize=11, fontweight='bold', color=color)
ax1.text(-0.65, -0.7, 'Wald (β_F=β_I):', ha='right', va='top',
         fontsize=8.5, style='italic', color='#444')

ax1.axhline(y=11.83, color=COLOR_F, lw=1.1, ls='--',
            alpha=0.55, zorder=2, label='Ret. formal global (11.83%)')
ax1.axhline(y=5.03, color=COLOR_SIG, lw=1.1, ls='--',
            alpha=0.55, zorder=2, label='Ret. informal global (5.03%)')

ymax = max(max(ret_f), max(ret_i)) + max(max(se_f), max(se_i))*1.96 + 2.5
ax1.set_ylim(-1.5, ymax)
ax1.set_xticks(x)
ax1.set_xticklabels(etiq_x, fontsize=10)
ax1.set_ylabel('Retorno a la escolaridad (%)\npor año adicional', fontsize=10)
ax1.grid(axis='y', alpha=0.3, zorder=1)
ax1.set_axisbelow(True)
ax1.legend(loc='upper left', fontsize=8.5, framealpha=0.92, ncol=2)
ax1.set_title(
    'Figura 10. Retornos a la escolaridad por bloque SCIAN\n'
    'Sector terciario privado, CDMX y Estado de México, 2005–2025\n'
    '(Barras de error: intervalos de confianza al 95%)',
    fontsize=12, fontweight='bold', pad=10
)

# ── Panel inferior — Ratio F/I ──
col_bars = [COLOR_SIG if p < 0.05 else COLOR_NS for p in pvals]
ax2.bar(x, ratios, 0.5, color=col_bars, zorder=3,
        edgecolor='#333', linewidth=0.8)
for xi, ratio, p in zip(x, ratios, pvals):
    ax2.text(xi, ratio + 0.08, f'{ratio:.2f}x',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.axhline(y=1,    color='#888', lw=1.0, ls='-',  alpha=0.4, zorder=2)
ax2.axhline(y=2.35, color='#444', lw=1.1, ls='--', alpha=0.6, zorder=2,
            label='Ratio global (2.35x)')
ax2.set_xticks(x)
ax2.set_xticklabels(etiq_x, fontsize=10)
ax2.set_ylim(0, 3.6)
ax2.set_ylabel('Ratio F / I', fontsize=10)
ax2.grid(axis='y', alpha=0.3, zorder=1)
ax2.set_axisbelow(True)
ax2.set_title('Ratio de retornos educativos formal / informal',
              fontsize=10.5, fontweight='bold')

red_patch  = mpatches.Patch(color=COLOR_SIG, label='Diferencia significativa (p<0.05)')
gray_patch = mpatches.Patch(color=COLOR_NS, label='Diferencia no significativa')
ax2.legend(handles=[red_patch, gray_patch], loc='upper right',
           fontsize=8.5, framealpha=0.92)

plt.savefig('FIGURA_10_retornos_bloque_v5.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Figura 10 guardada.\n")

# =============================================================================
# 7. FIGURA 11 — DESCOMPOSICIÓN OB
#    [3] Global reportado como 82.7% + 17.3% para comparabilidad visual
#    [2] Nomenclatura precisa en leyenda
#    [7] Nota al pie sin el desglose ficticio 17.4% + 1.7%
# =============================================================================
print("🎨 Generando Figura 11...")

b_ob    = [b for b in BLOQUES if b in ob_res]
expl_v  = [ob_res[b]['pct_e']  for b in b_ob]
nexpl_v = [ob_res[b]['pct_ne'] for b in b_ob]
brechas = [ob_res[b]['brecha'] for b in b_ob]

# [3] Global reportado como 82.7% + 17.3% para comparabilidad
b_ob_labels = [etiq_full[b] for b in b_ob] + ['GLOBAL\n(referencia)']
expl_v.append(GLOBAL_EXPL)
nexpl_v.append(GLOBAL_NEXPL_AGG)
brechas.append(GLOBAL_BRECHA)

expl_clip  = [max(v, 0) for v in expl_v]
nexpl_clip = [max(v, 0) for v in nexpl_v]

fig, ax = plt.subplots(figsize=(13, 6.8), constrained_layout=True)
xob = np.arange(len(b_ob_labels))
aw  = 0.55

COLOR_EXPL  = '#8fa8c8'
COLOR_NEXPL = '#1a2e4a'

ax.bar(xob, expl_clip, aw,
       label='Componente explicado\n(Dotaciones de capital humano observable)',
       color=COLOR_EXPL, zorder=3)
ax.bar(xob, nexpl_clip, aw, bottom=expl_clip,
       label='Componente no explicado\n(Estructural / segmentación residual)',
       color=COLOR_NEXPL, zorder=3)

# Etiquetas dentro de las barras
for i, (e, ne, b) in enumerate(zip(expl_v, nexpl_v, brechas)):
    if abs(e) > 4:
        ax.text(i, e/2, f'{e:.1f}%', ha='center', va='center',
                fontsize=10.5, color=COLOR_NEXPL, fontweight='bold')
    if abs(ne) > 4:
        bottom = max(e, 0)
        ax.text(i, bottom + ne/2, f'{ne:.1f}%', ha='center', va='center',
                fontsize=10.5, color='white', fontweight='bold')
    total_bar = max(e, 0) + max(ne, 0)
    ax.text(i, total_bar + 1.5, f'Δ={b:.3f}',
            ha='center', va='bottom', fontsize=8.5,
            color='#444', style='italic')

# Nota especial para Masivos
idx_m = ([etiq_full[b] for b in b_ob].index('Servicios Masivos')
         if 'Servicios Masivos' in [etiq_full[b] for b in b_ob] else None)
if idx_m is not None and abs(nexpl_v[idx_m]) < 5:
    ax.annotate(
        f'Comp. no expl.: {nexpl_v[idx_m]:.1f}%\n(prácticamente nulo)',
        xy=(idx_m, 98), xytext=(idx_m + 0.7, 80),
        fontsize=8, color='#c0392b',
        arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.1),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='#c0392b', alpha=0.9)
    )

# Línea de 100% como referencia
ax.axhline(y=100, color='#333', lw=0.8, ls=':', alpha=0.4, zorder=1)

ax.set_xticks(xob)
ax.set_xticklabels(b_ob_labels, fontsize=10)
ax.set_ylabel('Proporción de la brecha salarial (%)', fontsize=10)
ax.set_ylim(0, 115)
ax.grid(axis='y', alpha=0.25, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc='upper right', fontsize=9, framealpha=0.92)
ax.set_title(
    'Figura 11. Descomposición Oaxaca-Blinder por bloque SCIAN\n'
    'Sector terciario privado, CDMX y Estado de México, 2005–2025\n'
    'Referencia: coeficientes del segmento formal. Ponderado por factor ENOE.',
    fontsize=12, fontweight='bold', pad=10
)

# [7] Nota al pie sin el desglose ficticio 17.4% + 1.7%
ax.text(0.01, -0.10,
        'Nota: Descomposición ponderada por factor de expansión ENOE. '
        'Referencia: coeficientes del segmento formal.\n'
        'Para Servicios Masivos, el componente no explicado es '
        'prácticamente cero por cancelación entre retornos e intercepto.',
        transform=ax.transAxes, fontsize=7.5, color='#555',
        verticalalignment='top', wrap=True)

plt.savefig('FIGURA_11_oaxaca_bloque_v5.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Figura 11 guardada.\n")

# =============================================================================
# 8. NARRATIVAS PARA CAPÍTULO 5 — NOMENCLATURA CORREGIDA
#    ⚠️ VERIFICAR: los porcentajes escritos abajo deben coincidir con la
#       Tabla 8 impresa (ob_res). Si no coinciden, actualízalos aquí y en
#       el Capítulo 5. Se imprime una comprobación automática al final.
# =============================================================================
print("="*78)
print("RESUMEN INTERPRETATIVO — NARRATIVA PARA CAPÍTULO 5 Y 6")
print("="*78)

narrativas = {
    'B1_Comercio': (
        "El comercio presenta retornos educativos formales 2.40 veces mayores que "
        "los informales (9.28% vs 3.86%; Wald=585.6, p<0.001). La brecha salarial "
        "se explica en un 52.4% por diferencias en dotaciones observables de capital "
        "humano y en un 47.6% por el componente no explicado, compatible con "
        "diferencias estructurales en la remuneración de características observables."
    ),
    'B2_Alta_Cal': (
        "En servicios de alta calificación, los retornos formales son superiores "
        "(12.74% vs 9.86%; Wald=56.6, p<0.001), aunque la diferencia es moderada "
        "(ratio 1.29x). El componente no explicado representa 60.5% de la brecha, "
        "el más alto de los cuatro bloques, lo que indica que la segmentación "
        "opera más por mecanismos asociados a la remuneración diferenciada que "
        "por diferencias en dotaciones observables de capital humano."
    ),
    'B3_Masivos': (
        "Servicios Masivos registra la segmentación más severa en retornos educativos "
        "(8.42% vs 2.93%; Wald=1224.5, p<0.001; ratio 2.87x). Sin embargo, la "
        "descomposición Oaxaca-Blinder indica que prácticamente toda la brecha se "
        "atribuye a diferencias en dotaciones observables de capital humano (100.2%), "
        "con un componente no explicado prácticamente nulo. Esto refleja que los "
        "efectos de retornos diferenciales y de intercepto operan en direcciones "
        "opuestas, cancelándose mutuamente en el agregado del bloque."
    ),
    'B4_Sociales': (
        "En servicios sociales los retornos educativos son estadísticamente equivalentes "
        "entre segmentos (10.29% vs 10.03%; Wald=0.64, p=0.42, n.s.). No obstante, "
        "la brecha salarial total es la más amplia de los cuatro bloques (Δ=0.413). "
        "El componente no explicado representa 55.9% de dicha brecha, lo que indica "
        "que la segmentación opera en este bloque principalmente por selección "
        "ocupacional y arreglos contractuales, no por diferencias en retornos educativos."
    ),
}

for bloque in BLOQUES:
    print(f"\n► {etiq_full[bloque]}")
    print(f"  {narrativas[bloque]}")

# ── COMPROBACIÓN AUTOMÁTICA: narrativa vs cálculo ──────────────────────────
# Extrae el % no explicado que quedó escrito a mano en cada narrativa y lo
# compara con ob_res. Si difiere en más de 0.5 pp, avisa. Evita el drift
# entre lo que el script calcula y lo que la narrativa afirma.
import re
print("\n" + "─"*78)
print("COMPROBACIÓN: % no explicado escrito en narrativa vs calculado")
print("─"*78)
hardcode_ne = {  # % no explicado citado a mano en cada narrativa (ajústalo si cambias el texto)
    'B1_Comercio': 47.6, 'B2_Alta_Cal': 60.5,
    'B3_Masivos':  -0.2, 'B4_Sociales': 55.9,
}
for b in BLOQUES:
    if b in ob_res:
        calc = ob_res[b]['pct_ne']
        dicho = hardcode_ne[b]
        flag = "OK" if abs(calc - dicho) <= 0.5 else "⚠️ REVISAR"
        print(f"  {etiq_full[b]:<30} narrativa={dicho:>6.1f}%  "
              f"calculado={calc:>6.1f}%  {flag}")
print()

print("="*78)
print("✅ Análisis v5 completado. Versión final lista para Capítulo 5.")
print("="*78 + "\n")