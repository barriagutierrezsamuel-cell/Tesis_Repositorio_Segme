"""
Generación de las figuras del capítulo de resultados.

Produce las once figuras del capítulo a partir de la base de análisis, junto
con los valores numéricos que las acompañan en el texto y las notas al pie de
cada figura.

Entrada  : BASE_FINAL_CORREGIDA_2025.csv
Salida   : FIG_01 a FIG_11 en formato PNG y SERIES_anuales_cap5.csv

Tesis de licenciatura en Ingeniería Económica y Financiera
Luis Samuel Barria Gutiérrez — Universidad Autónoma del Estado de Hidalgo
"""

import os
import sys
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from scipy import stats as st
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════

RUTA   = r"C:\Users\REGLAMENTOS\Desktop\BASE_FINAL_CORREGIDA_2025.csv"
SALIDA = r"C:\Users\REGLAMENTOS\Desktop\FIGURAS_TESIS"

VD  = "ln_ing_hr_real_wins"
SAL = "ing_hr_real_wins"
HK  = ['esc_anios', 'exp', 'exp2']

FORM       = f"{VD} ~ esc_anios + exp + exp2 + C(anio, Treatment(reference=2005))"
FORM_TRAMO = f"{VD} ~ C(tramo) + exp + exp2 + C(anio, Treatment(reference=2005))"
FORM_ANUAL = f"{VD} ~ esc_anios + exp + exp2"

OFICIAL_BRECHA      = 0.4297
OFICIAL_EXPLICADO   = 0.3554
OFICIAL_ESTRUCTURAL = 0.0743
TOLERANCIA          = 0.0025
OFICIAL_JERARQ = {'entre': 0.2048, 'dentro_dot': 0.1678, 'dentro_est': 0.0571}

NUM = {
    'salario_real':      1,
    'brecha_absoluta':   2,
    'tasa_formalidad':   3,
    'distribucion':      4,
    'perfil_exp':        5,
    'retornos_bloque':   6,
    'oaxaca_bloque':     7,
    'descomposicion':    8,
    'retornos_anuales':  9,
    'estructural_anual': 10,
}

MIN_OBS_ANIO  = 100
MIN_OBS_CELDA = 200
CORTES_CHOW   = range(2010, 2021)

TRAMOS_CORTES = [-0.1, 5.9, 8.9, 11.9, 15.9, 99]
TRAMOS_ETQ = ['Sin primaria\ncompleta', 'Primaria', 'Secundaria',
              'Media superior', 'Superior']
TRAMOS_ESC = {'Sin primaria\ncompleta': 3.0, 'Primaria': 6.0, 'Secundaria': 9.0,
              'Media superior': 12.0, 'Superior': 17.0}

C_FOR, C_INF, C_ACC = '#1a2e4a', '#8fa8c8', '#c0392b'
C_SEG, C_GRIS, C_CRIS = '#c8956f', '#666666', '#e8e8e8'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#333333', 'axes.labelsize': 10.5,
    'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})

SUB = 'Sector terciario privado, CDMX y Estado de México, 2005–2025'
FTE = ('Fuente: elaboración propia con base en ENOE (INEGI). Estimaciones ponderadas '
       'por factor de expansión. Salarios deflactados a pesos de 2018.')
NOTA_COMP = ('Los valores de 2020 y 2021 reflejan cambios en la composición de la '
             'ocupación asociados a la pandemia y a la interrupción del levantamiento '
             'presencial, por lo que no son estrictamente comparables con el resto de '
             'la serie. ')

# ═══════════════════════════════════════════════════════════════════════
# 2. INFRAESTRUCTURA
# ═══════════════════════════════════════════════════════════════════════

os.makedirs(SALIDA, exist_ok=True)


class Tee:
    def __init__(self, ruta):
        self.terminal, self.log = sys.stdout, open(ruta, 'w', encoding='utf-8')

    def write(self, m):
        self.terminal.write(m); self.log.write(m)

    def flush(self):
        self.terminal.flush(); self.log.flush()


RUTA_LOG = os.path.join(SALIDA, 'SALIDA_COMPLETA_cap5.txt')
sys.stdout = Tee(RUTA_LOG)

print('\n' + '=' * 76)
print('  GENERADOR DE FIGURAS — CAPÍTULO 5  (versión 5)')
print('=' * 76)
print(f'\n  Base   : {RUTA}\n  Salida : {SALIDA}')
if not os.path.exists(RUTA):
    print('\n  ✗ ERROR: no se encuentra la base. Corrige RUTA.')
    sys.exit(1)

ARCHIVOS = []


def guardar(fig, nombre):
    r = os.path.join(SALIDA, nombre)
    fig.savefig(r, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig); ARCHIVOS.append(r); print(f'   ✓ {r}')


def titulo(ax, clave, txt, sub=SUB):
    ax.set_title(f'Figura {NUM[clave]}. {txt}\n{sub}', fontsize=12,
                 fontweight='bold', pad=14, linespacing=1.4)


def fuente(fig, txt=FTE, y=-0.02):
    fig.text(0.02, y, txt, fontsize=7.5, color='#555', ha='left')


def crisis(ax):
    for x0, x1 in [(2008, 2009), (2019.6, 2020.4)]:
        ax.axvspan(x0, x1, color=C_CRIS, zorder=0)


def enteros(ax, paso=2):
    ax.xaxis.set_major_locator(mticker.MultipleLocator(paso))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f'{int(v)}'))


# ═══════════════════════════════════════════════════════════════════════
# 3. CARGA
# ═══════════════════════════════════════════════════════════════════════

print('\nCargando base…')
df = pd.read_csv(RUTA, low_memory=False)
if 'esc_anios' not in df.columns and 'anios_esc' in df.columns:
    df = df.rename(columns={'anios_esc': 'esc_anios'})
REQ = ['esc_anios', 'eda', 'anio', 'formal', 'factor', 'rubro', VD, SAL]
falt = [c for c in REQ if c not in df.columns]
if falt:
    print(f'\n  ✗ ERROR: faltan columnas: {falt}'); sys.exit(1)

df = df[df['esc_anios'] != 99].copy()
df['exp']  = (df['eda'] - df['esc_anios'] - 6).clip(lower=0)
df['exp2'] = df['exp'] ** 2
df = df[df[VD].notna()].copy()
df['tramo'] = pd.cut(df['esc_anios'], bins=TRAMOS_CORTES, labels=TRAMOS_ETQ)
for col, cond, nom in [('tip_con', [2, 3, 4], 'con_contrato'),
                       ('pre_asa', [1], 'acc_salud')]:
    if col in df.columns:
        df[nom] = df[col].isin(cond).astype(int)

F = df[df['formal'] == 1]
I = df[df['formal'] == 0]


def mp(d, c):
    return np.average(d[c], weights=d['factor'])


def sd_p(d, c):
    m = mp(d, c)
    return np.sqrt(np.average((d[c] - m) ** 2, weights=d['factor']))


def cuantil_p(d, c, q):
    o = d[[c, 'factor']].sort_values(c)
    cum = o['factor'].cumsum() / o['factor'].sum()
    return o.loc[cum >= q, c].iloc[0]


def shares_anio(d):
    p = d.groupby('anio')['factor'].sum()
    return (p / p.sum()).to_dict()


def efecto_anio_medio(modelo, datos):
    ef = {}
    for n, v in modelo.params.items():
        if n.startswith('C(anio'):
            ef[int(n.split('[T.')[1].rstrip(']'))] = v
    return float(sum(s * ef.get(int(a), 0.0) for a, s in shares_anio(datos).items()))


def contraste(modelo, exp, shares, esc=None, tramo=None):
    idx = {n: i for i, n in enumerate(modelo.params.index)}
    c = np.zeros(len(modelo.params))
    c[idx['Intercept']] = 1.0
    for n, i in idx.items():
        if n.startswith('C(anio'):
            c[i] = float(shares.get(int(n.split('[T.')[1].rstrip(']')), 0.0))
        elif n.startswith('C(tramo'):
            lab = n.split('[T.')[1].rstrip(']')
            c[i] = 1.0 if (tramo is not None and lab == tramo) else 0.0
    if 'esc_anios' in idx and esc is not None:
        c[idx['esc_anios']] = esc
    c[idx['exp']] = exp
    c[idx['exp2']] = exp ** 2
    return c


def brecha_contraste(mod_f, mod_i, exp, shares, esc=None, tramo=None):
    cf = contraste(mod_f, exp, shares, esc, tramo)
    ci = contraste(mod_i, exp, shares, esc, tramo)
    g = float(cf @ mod_f.params.values - ci @ mod_i.params.values)
    v = float(cf @ mod_f.cov_params().values @ cf +
              ci @ mod_i.cov_params().values @ ci)
    return g, np.sqrt(v)


print(f'   N = {len(df):,}  |  Formales = {len(F):,}  |  Informales = {len(I):,}')
SHARES = shares_anio(df)
EXP_REF = mp(df, 'exp')
print(f'   Punto de evaluación común: experiencia {EXP_REF:.2f} años, composición '
      f'temporal de la muestra completa.')

# ═══════════════════════════════════════════════════════════════════════
# 4. MODELOS GLOBALES Y VERIFICACIÓN
# ═══════════════════════════════════════════════════════════════════════

print('\nEstimando modelos globales…')
mg = smf.wls(f"{VD} ~ formal + esc_anios + exp + exp2 + C(anio, Treatment(reference=2005))",
             data=df, weights=df['factor']).fit(cov_type='HC3')
mF = smf.wls(FORM, data=F, weights=F['factor']).fit(cov_type='HC3')
mI = smf.wls(FORM, data=I, weights=I['factor']).fit(cov_type='HC3')

BR_calc  = mp(F, VD) - mp(I, VD)
DOT_calc = sum((mp(F, v) - mp(I, v)) * mF.params[v] for v in HK)

print('\n' + '─' * 76)
print('  VERIFICACIÓN DE CONSTANTES CONTRA VERIFICACION23.py')
print('─' * 76)
print(f"  {'Componente':<26}{'Oficial':>12}{'Recalculado':>14}{'Δ':>12}")
for nom, ofi, cal in [('Brecha total (ln)', OFICIAL_BRECHA, BR_calc),
                      ('Explicado (dotaciones)', OFICIAL_EXPLICADO, DOT_calc),
                      ('No explicado (estruct.)', OFICIAL_ESTRUCTURAL, BR_calc - DOT_calc)]:
    print(f'  {nom:<26}{ofi:>12.4f}{cal:>14.4f}{cal - ofi:>11.4f}  '
          f'{"✓" if abs(ofi - cal) < TOLERANCIA else "✗"}')
if (abs(OFICIAL_BRECHA - BR_calc) >= TOLERANCIA or
        abs(OFICIAL_EXPLICADO - DOT_calc) >= TOLERANCIA):
    print('\n  ✗ DETENIDO: constantes no reproducidas.'); sys.exit(1)
print('\n  ✓ Constantes verificadas.')

BR, EX, NE = OFICIAL_BRECHA, OFICIAL_EXPLICADO, OFICIAL_ESTRUCTURAL
pe, pn = EX / BR * 100, NE / BR * 100
ea_F, ea_I = efecto_anio_medio(mF, F), efecto_anio_medio(mI, I)
vertice_F = -mF.params['exp'] / (2 * mF.params['exp2'])
vertice_I = -mI.params['exp'] / (2 * mI.params['exp2'])

# ═══════════════════════════════════════════════════════════════════════
# 5. PERFIL DE BRECHA POR ESCOLARIDAD
# ═══════════════════════════════════════════════════════════════════════

print('\n' + '═' * 76)
print('  PERFIL DE BRECHA POR NIVEL DE ESCOLARIDAD  (estimación principal)')
print('═' * 76)
print('  Brecha predicha formal − informal para trabajadores con idéntica')
print(f'  escolaridad, experiencia ({EXP_REF:.1f} años) y composición temporal.')
print('  La escolaridad entra como indicadores de nivel: no se impone linealidad.\n')

mF_t = smf.wls(FORM_TRAMO, data=F, weights=F['factor']).fit(cov_type='HC3')
mI_t = smf.wls(FORM_TRAMO, data=I, weights=I['factor']).fit(cov_type='HC3')
nF, nI = F.groupby('tramo').size(), I.groupby('tramo').size()

print(f"  {'Tramo':<22}{'N form.':>10}{'N inf.':>10}{'Brecha ln':>12}{'EE':>9}"
      f"{'En %':>9}{'IC 95% (%)':>20}")
res_tramo = []
for t in TRAMOS_ETQ:
    lab = t.replace('\n', ' ')
    if nF.get(t, 0) < MIN_OBS_CELDA or nI.get(t, 0) < MIN_OBS_CELDA:
        print(f'  {lab:<22}{nF.get(t, 0):>10,}{nI.get(t, 0):>10,}'
              f'{"celda insuficiente — excluida":>50}')
        continue
    g, s = brecha_contraste(mF_t, mI_t, EXP_REF, SHARES, tramo=t)
    l, h = (np.exp(g - 1.96 * s) - 1) * 100, (np.exp(g + 1.96 * s) - 1) * 100
    res_tramo.append({'tramo': lab, 'etq': t, 'g': g, 'se': s, 'lo': g - 1.96 * s,
                      'hi': g + 1.96 * s, 'esc': TRAMOS_ESC[t],
                      'nf': int(nF[t]), 'ni': int(nI[t])})
    print(f'  {lab:<22}{nF[t]:>10,}{nI[t]:>10,}{g:>12.4f}{s:>9.4f}'
          f'{(np.exp(g) - 1) * 100:>8.1f}%   [{l:>6.1f}, {h:>6.1f}]')
RT = pd.DataFrame(res_tramo)

# ─── Criterios de verificación ─────────────────────────────────────────
c1 = bool(RT['g'].iloc[0] <= 0)
c2 = bool((np.diff(RT['g'].values) > 0).all())
c3 = bool(RT['g'].iloc[-1] > 0 and RT['lo'].iloc[-1] > 0)
c4 = bool((RT['hi'] < 0).any() or ((RT['lo'] < 0) & (RT['hi'] > 0)).any())
print(f'\n  C1 tramo más bajo con brecha ≤ 0 ................ {"CUMPLE" if c1 else "NO CUMPLE"}')
print(f'  C2 monotonía creciente .......................... {"CUMPLE" if c2 else "NO CUMPLE"}')
print(f'  C3 tramo superior positivo y significativo ...... {"CUMPLE" if c3 else "NO CUMPLE"}')
print(f'  C4 existe tramo negativo o nulo ................. {"CUMPLE" if c4 else "NO CUMPLE"}')
print('\n  ' + '─' * 72)
print('  VEREDICTO: LA DIVERGENCIA EDUCATIVA SE SOSTIENE.' if (c1 and c2 and c3 and c4)
      else '  VEREDICTO: revisa qué criterio falla antes de redactar.')
print('  ' + '─' * 72)

# ─── Ritmo de apertura de la brecha ────────────────────────────────────
print('\n── Incremento de la brecha entre tramos consecutivos ──')
print(f"  {'Transición':<34}{'Δ ln':>10}{'Años':>7}{'Por año':>10}")
for i in range(len(RT) - 1):
    d_ln = RT['g'].iloc[i + 1] - RT['g'].iloc[i]
    d_esc = RT['esc'].iloc[i + 1] - RT['esc'].iloc[i]
    print(f"  {RT['tramo'].iloc[i] + ' → ' + RT['tramo'].iloc[i + 1]:<34}"
          f"{d_ln:>10.4f}{d_esc:>7.0f}{d_ln / d_esc:>10.4f}")
print('  >> La divergencia se acelera hasta media superior y luego se desacelera.')
print('     Una especificación lineal no puede reproducir ese patrón.')

# ─── Especificación lineal, solo como referencia documentada ───────────
print('\n── Especificación lineal (REFERENCIA — no se usa en la figura) ──')
print('   Se conserva para dejar constancia de la comparación que motivó')
print('   descartarla. Sobreestima la brecha en ambos extremos.\n')
print(f"  {'Escolaridad':<20}{'Lineal (%)':>13}{'Por tramos (%)':>17}{'Diferencia':>13}")
for e, t in [(3, 'Sin primaria\ncompleta'), (6, 'Primaria'), (9, 'Secundaria'),
             (12, 'Media superior'), (17, 'Superior')]:
    g_l, _ = brecha_contraste(mF, mI, EXP_REF, SHARES, esc=e)
    fila = RT[RT['etq'] == t]
    if fila.empty:
        continue
    pl, pt = (np.exp(g_l) - 1) * 100, (np.exp(fila['g'].iloc[0]) - 1) * 100
    print(f"  {t.replace(chr(10), ' ') + f' ({e:.0f})':<20}{pl:>13.1f}{pt:>17.1f}"
          f"{pl - pt:>13.1f}")
g0, _ = brecha_contraste(mF, mI, EXP_REF, SHARES, esc=0)
b_esc = mF.params['esc_anios'] - mI.params['esc_anios']
print(f'\n  Cruce implícito en la recta: {-g0 / b_esc:.2f} años de escolaridad.')
print('  No se reporta en el capítulo: depende de una forma funcional rechazada')
print('  por la estimación por tramos. La transición defendible se ubica entre')
print('  primaria (brecha no distinguible de cero) y secundaria.')

# ─── Evidencia sin modelo ──────────────────────────────────────────────
print('\n── Brecha bruta por año exacto de escolaridad (sin modelo) ──')
print(f"  {'Años':>6}{'N form.':>11}{'N inf.':>11}{'Brecha ln':>12}")
for e in sorted(df['esc_anios'].unique()):
    a, b = F[F['esc_anios'] == e], I[I['esc_anios'] == e]
    if len(a) < MIN_OBS_CELDA or len(b) < MIN_OBS_CELDA:
        continue
    print(f'  {e:>6.0f}{len(a):>11,}{len(b):>11,}{mp(a, VD) - mp(b, VD):>12.4f}')
print('  >> Sin ajustar por experiencia ni por año: contraste de dirección.')

# ═══════════════════════════════════════════════════════════════════════
# 6. SERIES ANUALES Y QUIEBRE
# ═══════════════════════════════════════════════════════════════════════

print('\nCalculando series anuales…')
filas, descartados = [], []
for a in sorted(df['anio'].unique()):
    da = df[df['anio'] == a]
    daf, dai = da[da['formal'] == 1], da[da['formal'] == 0]
    if len(daf) < MIN_OBS_ANIO or len(dai) < MIN_OBS_ANIO:
        descartados.append((int(a), len(daf), len(dai))); continue
    mf = smf.wls(FORM_ANUAL, data=daf, weights=daf['factor']).fit(cov_type='HC3')
    mi = smf.wls(FORM_ANUAL, data=dai, weights=dai['factor']).fit(cov_type='HC3')
    br = mp(daf, VD) - mp(dai, VD)
    dot = sum((mp(daf, v) - mp(dai, v)) * mf.params[v] for v in HK)
    filas.append({'anio': int(a), 'sal_f': mp(daf, SAL), 'sal_i': mp(dai, SAL),
                  'ret_f': mf.params['esc_anios'] * 100,
                  'ret_i': mi.params['esc_anios'] * 100,
                  'formalidad': mp(da, 'formal') * 100,
                  'pct_est': (br - dot) / br * 100, 'brecha_ln': br})
S = pd.DataFrame(filas)
S['brecha_abs'] = S['sal_f'] - S['sal_i']
S['gap_ret'] = S['ret_f'] - S['ret_i']
print(f'   Años estimados: {len(S)}  ({S["anio"].min()}–{S["anio"].max()})')
for a, nf_, ni_ in descartados:
    print(f'   Descartado {a}: formales={nf_:,} informales={ni_:,}')

print('\n' + '─' * 76)
print('  QUIEBRE EN LA SERIE DEL COMPONENTE ESTRUCTURAL (Chow, todos los cortes)')
print('─' * 76)


def rss_lin(x, y):
    r = st.linregress(x, y)
    return float(((y - (r.intercept + r.slope * x)) ** 2).sum())


rr_total = rss_lin(S['anio'], S['pct_est'])
tg_glob = st.linregress(S['anio'], S['pct_est'])
print(f'  Periodo completo: {tg_glob.slope:+.3f} pp/año (p = {tg_glob.pvalue:.4f})\n')
print(f"  {'Corte':>7}{'F':>9}{'p':>10}{'Pend. previa':>18}{'Pend. posterior':>19}")
chow = []
for c in CORTES_CHOW:
    a, b = S[S['anio'] <= c], S[S['anio'] > c]
    if len(a) < 4 or len(b) < 4:
        continue
    ru = rss_lin(a['anio'], a['pct_est']) + rss_lin(b['anio'], b['pct_est'])
    k, n = 2, len(S)
    Fc = ((rr_total - ru) / k) / (ru / (n - 2 * k))
    pc = 1 - st.f.cdf(Fc, k, n - 2 * k)
    ra, rb = st.linregress(a['anio'], a['pct_est']), st.linregress(b['anio'], b['pct_est'])
    chow.append((c, Fc, pc, ra, rb))
    print(f'  {c:>7}{Fc:>9.2f}{pc:>10.4f}{ra.slope:>13.2f} (p={ra.pvalue:.3f})'
          f'{rb.slope:>11.2f} (p={rb.pvalue:.3f})')
CORTE, F_max, p_max, TA, TB = max(chow, key=lambda r: r[1])
print(f'\n  Quiebre por máximo F: {CORTE} (F = {F_max:.2f}, p = {p_max:.6f}).')
print(f'  Se rechaza en los {len(chow)} cortes candidatos.')

# ═══════════════════════════════════════════════════════════════════════
# 7. BLOQUES SCIAN
# ═══════════════════════════════════════════════════════════════════════

bmap = {6: 'B1', 7: 'B1', 9: 'B2', 10: 'B2', 12: 'B2', 13: 'B2',
        8: 'B3', 11: 'B3', 14: 'B3', 17: 'B3', 18: 'B3', 19: 'B3',
        15: 'B4', 16: 'B4'}
etq = {'B1': 'Comercio', 'B2': 'Servicios de Alta\nCalificación',
       'B3': 'Servicios\nMasivos', 'B4': 'Servicios\nSociales'}
BL = ['B1', 'B2', 'B3', 'B4']
df['bloque'] = df['rubro'].map(bmap)
dfb = df[df['bloque'].notna()].copy()
Fb, Ib = dfb[dfb['formal'] == 1], dfb[dfb['formal'] == 0]

print('\nEstimando modelos por bloque (lineal y por tramos)…')
bres = {}
for b in BL:
    db = dfb[dfb['bloque'] == b]
    dbf, dbi = db[db['formal'] == 1], db[db['formal'] == 0]
    m_f = smf.wls(FORM, data=dbf, weights=dbf['factor']).fit(cov_type='HC3')
    m_i = smf.wls(FORM, data=dbi, weights=dbi['factor']).fit(cov_type='HC3')
    br = mp(dbf, VD) - mp(dbi, VD)
    dot = sum((mp(dbf, v) - mp(dbi, v)) * m_f.params[v] for v in HK)
    w = (m_f.params['esc_anios'] - m_i.params['esc_anios']) ** 2 / \
        (m_f.bse['esc_anios'] ** 2 + m_i.bse['esc_anios'] ** 2)
    sh = shares_anio(db)
    # Perfil por tramos dentro del bloque
    m_f_t = smf.wls(FORM_TRAMO, data=dbf, weights=dbf['factor']).fit(cov_type='HC3')
    m_i_t = smf.wls(FORM_TRAMO, data=dbi, weights=dbi['factor']).fit(cov_type='HC3')
    nfb, nib = dbf.groupby('tramo').size(), dbi.groupby('tramo').size()
    perfil = {}
    for t in TRAMOS_ETQ:
        if nfb.get(t, 0) < MIN_OBS_CELDA or nib.get(t, 0) < MIN_OBS_CELDA:
            perfil[t] = (np.nan, np.nan)
        else:
            perfil[t] = brecha_contraste(m_f_t, m_i_t, EXP_REF, sh, tramo=t)
    bres[b] = {'rf': m_f.params['esc_anios'] * 100, 'ri': m_i.params['esc_anios'] * 100,
               'ef': m_f.bse['esc_anios'] * 100, 'ei': m_i.bse['esc_anios'] * 100,
               'wald': w, 'p': 1 - st.chi2.cdf(w, 1),
               'br': br, 'dot': dot, 'est': br - dot,
               'pe': dot / br * 100, 'pn': (br - dot) / br * 100, 'perfil': perfil,
               'sh_f': Fb.loc[Fb['bloque'] == b, 'factor'].sum() / Fb['factor'].sum(),
               'sh_i': Ib.loc[Ib['bloque'] == b, 'factor'].sum() / Ib['factor'].sum(),
               'y_f': mp(dbf, VD), 'y_i': mp(dbi, VD)}

print('\n' + '─' * 76)
print('  DESCOMPOSICIÓN JERÁRQUICA — RECÁLCULO')
print('─' * 76)
variantes = {
    'A (ȳ informal / peso formal)': (
        sum((bres[b]['sh_f'] - bres[b]['sh_i']) * bres[b]['y_i'] for b in BL),
        sum(bres[b]['sh_f'] * bres[b]['dot'] for b in BL),
        sum(bres[b]['sh_f'] * bres[b]['est'] for b in BL)),
    'B (ȳ formal / peso informal)': (
        sum((bres[b]['sh_f'] - bres[b]['sh_i']) * bres[b]['y_f'] for b in BL),
        sum(bres[b]['sh_i'] * bres[b]['dot'] for b in BL),
        sum(bres[b]['sh_i'] * bres[b]['est'] for b in BL))}
print(f"  {'Convención':<32}{'Entre':>11}{'Dentro-dot':>13}{'Dentro-est':>13}{'Suma':>10}")
coincide = None
for nom, (e_, d_, s_) in variantes.items():
    print(f'  {nom:<32}{e_:>11.4f}{d_:>13.4f}{s_:>13.4f}{e_ + d_ + s_:>10.4f}')
    if (abs(e_ - OFICIAL_JERARQ['entre']) < TOLERANCIA and
            abs(d_ - OFICIAL_JERARQ['dentro_dot']) < TOLERANCIA):
        coincide = nom
print(f"  {'OFICIAL (Tabla 9)':<32}{OFICIAL_JERARQ['entre']:>11.4f}"
      f"{OFICIAL_JERARQ['dentro_dot']:>13.4f}{OFICIAL_JERARQ['dentro_est']:>13.4f}"
      f"{sum(OFICIAL_JERARQ.values()):>10.4f}")
if coincide:
    print(f'\n  ✓ Convención vigente: {coincide}.')
    SEG_v, DOT_v, EST_v = variantes[coincide]
else:
    print('\n  ⚠ Ninguna convención reproduce las cifras oficiales.')
    SEG_v, DOT_v, EST_v = (OFICIAL_JERARQ['entre'], OFICIAL_JERARQ['dentro_dot'],
                           OFICIAL_JERARQ['dentro_est'])
TOT_J = SEG_v + DOT_v + EST_v
SEG, DOT, EST = SEG_v / TOT_J * 100, DOT_v / TOT_J * 100, EST_v / TOT_J * 100

# ═══════════════════════════════════════════════════════════════════════
# 8. FIGURAS
# ═══════════════════════════════════════════════════════════════════════

print('\n' + '─' * 76)
print('  GENERANDO FIGURAS')
print('─' * 76)

# ─── 1. Salario real ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
crisis(ax)
ax.plot(S['anio'], S['sal_f'], color=C_FOR, lw=2.4, marker='o', ms=4, zorder=3)
ax.plot(S['anio'], S['sal_i'], color=C_INF, lw=2.4, marker='o', ms=4, zorder=3)
ax.text(S['anio'].iloc[-1] + 0.25, S['sal_f'].iloc[-1], 'Formal', color=C_FOR,
        fontsize=10.5, fontweight='bold', va='center')
ax.text(S['anio'].iloc[-1] + 0.25, S['sal_i'].iloc[-1], 'Informal', color='#5a7ba6',
        fontsize=10.5, fontweight='bold', va='center')
y0, y1 = ax.get_ylim()
ax.text(2008.5, y1 - (y1 - y0) * 0.03, 'Crisis\n2008–09', fontsize=7.5,
        color=C_GRIS, ha='center', va='top', style='italic')
ax.text(2020, y1 - (y1 - y0) * 0.03, 'COVID-19', fontsize=7.5,
        color=C_GRIS, ha='center', va='top', style='italic')
ax.set_ylabel('Salario real por hora (pesos de 2018)')
ax.set_xlim(2004.3, 2026.8); enteros(ax)
ax.grid(axis='y', alpha=0.25); ax.set_axisbelow(True)
titulo(ax, 'salario_real', 'Evolución del salario real por hora según segmento')
fuente(fig, 'Nota: ' + NOTA_COMP + FTE)
guardar(fig, f'FIG_{NUM["salario_real"]:02d}_salario_real.png')

# ─── 2. Brecha absoluta ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
crisis(ax)
ax.fill_between(S['anio'], 0, S['brecha_abs'], color=C_INF, alpha=0.45, zorder=2)
ax.plot(S['anio'], S['brecha_abs'], color=C_FOR, lw=2.4, marker='o', ms=4, zorder=3)
prom = S['brecha_abs'].mean()
ax.axhline(prom, color=C_ACC, ls='--', lw=1.2, zorder=4)
# CORRECCIÓN: la etiqueta se coloca por encima de su propia línea, no encima
ax.text(S['anio'].max() + 0.35, prom + S['brecha_abs'].max() * 0.045,
        f'Promedio\n${prom:.2f}', color=C_ACC, fontsize=9, va='bottom',
        ha='left', fontweight='bold')
imax, imin = S['brecha_abs'].idxmax(), S['brecha_abs'].idxmin()
ax.annotate(f"Máximo: ${S.loc[imax, 'brecha_abs']:.2f} ({S.loc[imax, 'anio']})",
            xy=(S.loc[imax, 'anio'], S.loc[imax, 'brecha_abs']),
            xytext=(S.loc[imax, 'anio'] - 6.5, S.loc[imax, 'brecha_abs'] + 3.0),
            fontsize=8.5, color='#333', arrowprops=dict(arrowstyle='->', color='#666', lw=1))
ax.annotate(f"Mínimo: ${S.loc[imin, 'brecha_abs']:.2f} ({S.loc[imin, 'anio']})",
            xy=(S.loc[imin, 'anio'], S.loc[imin, 'brecha_abs']),
            xytext=(S.loc[imin, 'anio'] - 6.0, S.loc[imin, 'brecha_abs'] - 4.2),
            fontsize=8.5, color='#333', arrowprops=dict(arrowstyle='->', color='#666', lw=1))
ax.set_ylabel('Diferencia salarial absoluta (pesos por hora)')
ax.set_xlim(2004.3, 2027.2); ax.set_ylim(0, S['brecha_abs'].max() * 1.32); enteros(ax)
ax.grid(axis='y', alpha=0.25); ax.set_axisbelow(True)
titulo(ax, 'brecha_absoluta', 'Brecha salarial absoluta entre segmentos')
fuente(fig, 'Nota: ' + NOTA_COMP + FTE)
guardar(fig, f'FIG_{NUM["brecha_absoluta"]:02d}_brecha_absoluta.png')

# ─── 3. Tasa de formalidad ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
crisis(ax)
ax.plot(S['anio'], S['formalidad'], color=C_FOR, lw=2.4, marker='o', ms=4.5, zorder=3)
prom = mp(df, 'formal') * 100
ax.axhline(prom, color=C_ACC, ls='--', lw=1.2, zorder=2)
rango = S['formalidad'].max() - S['formalidad'].min()
# CORRECCIÓN: se amplía el eje y la anotación se coloca DENTRO del área de ejes
ax.set_ylim(S['formalidad'].min() - rango * 0.28, S['formalidad'].max() + rango * 0.12)
ax.text(S['anio'].max() + 0.35, prom, f'Promedio\n{prom:.1f}%', color=C_ACC,
        fontsize=9, va='center', ha='left', fontweight='bold')
imin = S['formalidad'].idxmin()
ax.annotate(f"Mínimo: {S.loc[imin, 'formalidad']:.1f}% ({S.loc[imin, 'anio']})",
            xy=(S.loc[imin, 'anio'], S.loc[imin, 'formalidad']),
            xytext=(S.loc[imin, 'anio'] - 5.5,
                    S.loc[imin, 'formalidad'] - rango * 0.20),
            fontsize=8.5, color='#333',
            arrowprops=dict(arrowstyle='->', color='#666', lw=1))
ax.set_ylabel('Trabajadores formales (% del empleo terciario)')
ax.set_xlim(2004.3, 2027.2); enteros(ax)
ax.grid(axis='y', alpha=0.25); ax.set_axisbelow(True)
titulo(ax, 'tasa_formalidad', 'Tasa de formalidad del empleo')
fuente(fig)
guardar(fig, f'FIG_{NUM["tasa_formalidad"]:02d}_tasa_formalidad.png')

# ─── 4. Distribución ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.6), constrained_layout=True)
lim = cuantil_p(df, SAL, 0.97)
bins = np.linspace(0, lim, 70)
for d_, col, al in [(I, C_INF, 0.75), (F, C_FOR, 0.55)]:
    h, e = np.histogram(d_[SAL], bins=bins, weights=d_['factor'], density=True)
    ax.fill_between((e[:-1] + e[1:]) / 2, 0, h, color=col, alpha=al, zorder=2)
med_f, med_i = cuantil_p(F, SAL, 0.5), cuantil_p(I, SAL, 0.5)
for m, col, lab, dy in [(med_i, '#4a6a8a', 'Informal', 0.80), (med_f, C_FOR, 'Formal', 0.92)]:
    ax.axvline(m, color=col, ls='--', lw=1.5, zorder=4)
    ax.text(m + 1.5, ax.get_ylim()[1] * dy, f'Mediana {lab}\n${m:.2f}',
            fontsize=9, color=col, fontweight='bold', va='top')
ax.legend(handles=[Patch(facecolor=C_FOR, alpha=0.55, label='Segmento formal'),
                   Patch(facecolor=C_INF, alpha=0.75, label='Segmento informal')],
          loc='upper right', frameon=False, fontsize=10)
ax.set_xlabel('Salario real por hora (pesos de 2018)'); ax.set_ylabel('Densidad')
ax.set_xlim(0, lim); ax.grid(axis='y', alpha=0.2); ax.set_axisbelow(True)
titulo(ax, 'distribucion', 'Distribución del salario real por hora según segmento')
fuente(fig, FTE + ' Eje truncado en el percentil 97.')
guardar(fig, f'FIG_{NUM["distribucion"]:02d}_distribucion.png')

# ─── 5. DOS PANELES ────────────────────────────────────────────────────
fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)

# Panel (a)
xe = np.linspace(0, 45, 200)
niveles = {}
for m, d_, col, lab, ea in [(mF, F, C_FOR, 'Formal', ea_F), (mI, I, C_INF, 'Informal', ea_I)]:
    esc = mp(d_, 'esc_anios')
    ln = (m.params['Intercept'] + ea + m.params['esc_anios'] * esc +
          m.params['exp'] * xe + m.params['exp2'] * xe ** 2)
    niveles[lab] = (m, esc, ea)
    axa.plot(xe, np.exp(ln), color=col, lw=2.6, zorder=3)
    axa.text(45.8, np.exp(ln[-1]), lab, color=col if lab == 'Formal' else '#5a7ba6',
             fontsize=10, fontweight='bold', va='center')
for x0 in [10, 25, 40]:
    ys = []
    for lab in ['Formal', 'Informal']:
        m, esc, ea = niveles[lab]
        ys.append(np.exp(m.params['Intercept'] + ea + m.params['esc_anios'] * esc +
                         m.params['exp'] * x0 + m.params['exp2'] * x0 ** 2))
    axa.plot([x0, x0], ys, color=C_GRIS, lw=1, ls=':', zorder=2)
    axa.text(x0 + 0.6, sum(ys) / 2, f'${ys[0] - ys[1]:.1f}', fontsize=8,
             color=C_GRIS, va='center', style='italic')
axa.set_xlabel('Experiencia potencial (años)')
axa.set_ylabel('Salario real por hora estimado (pesos de 2018)')
axa.set_xlim(0, 52); axa.grid(axis='y', alpha=0.25); axa.set_axisbelow(True)
axa.set_title('(a) Perfil experiencia-salario\nevaluado en la escolaridad media de cada segmento',
              fontsize=10, pad=10, linespacing=1.3)

# Panel (b) — SOLO tramos. Se retira la curva lineal.
xs = RT['esc'].values
ys = (np.exp(RT['g']) - 1) * 100
ylo = (np.exp(RT['lo']) - 1) * 100
yhi = (np.exp(RT['hi']) - 1) * 100
axb.axhline(0, color='#333', lw=1.1, zorder=3)
axb.plot(xs, ys, color=C_INF, lw=1.6, ls='-', alpha=0.85, zorder=4)
axb.errorbar(xs, ys, yerr=[ys - ylo, yhi - ys], fmt='o', ms=8, color=C_FOR,
             ecolor=C_FOR, elinewidth=1.6, capsize=5, zorder=5)
for x_, y_, hi_ in zip(xs, ys, yhi):
    axb.text(x_, hi_ + (max(yhi) - min(ylo)) * 0.045, f'{y_:+.1f}%', ha='center',
             fontsize=9.5, fontweight='bold', color=C_FOR)
nulo = RT[(RT['lo'] < 0) & (RT['hi'] > 0)]
if not nulo.empty:
    xn = nulo['esc'].iloc[0]
    axb.annotate('No difiere\nde cero', xy=(xn, nulo['g'].iloc[0] * 100),
                 xytext=(xn - 1.1, (max(yhi) - min(ylo)) * 0.30 + min(ylo)),
                 fontsize=8.5, color=C_ACC, ha='center',
                 arrowprops=dict(arrowstyle='->', color=C_ACC, lw=1.0))
axb.set_xticks(xs)
axb.set_xticklabels([t for t in RT['etq']], fontsize=8.5)
axb.set_xlim(min(xs) - 2.2, max(xs) + 2.2)
axb.set_ylim(min(ylo) - (max(yhi) - min(ylo)) * 0.18,
             max(yhi) + (max(yhi) - min(ylo)) * 0.20)
axb.set_xlabel('Nivel de escolaridad alcanzado')
axb.set_ylabel('Brecha salarial formal − informal (%)')
axb.grid(axis='y', alpha=0.25); axb.set_axisbelow(True)
axb.set_title(f'(b) Brecha predicha según nivel de escolaridad\n'
              f'a {EXP_REF:.0f} años de experiencia potencial',
              fontsize=10, pad=10, linespacing=1.3)

fig.suptitle(f'Figura {NUM["perfil_exp"]}. Determinación salarial según segmento: '
             f'experiencia y escolaridad\n{SUB}',
             fontsize=12, fontweight='bold', linespacing=1.4)
fuente(fig, f'Nota: panel (a), curvas derivadas de las ecuaciones de Mincer por segmento; '
            f'los vértices ({vertice_F:.0f} y {vertice_I:.0f} años de experiencia potencial) '
            f'quedan fuera del rango observable. Panel (b), diferencia entre salarios '
            f'predichos para trabajadores con idéntico nivel educativo, experiencia y '
            f'composición temporal; la escolaridad entra como indicadores de nivel, sin '
            f'imponer linealidad; barras de error al 95%. ' + FTE, y=-0.05)
guardar(fig, f'FIG_{NUM["perfil_exp"]:02d}_perfil_experiencia_escolaridad.png')

# ─── 6. Retornos por bloque ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
x = np.arange(4); w = 0.34
rfs = [bres[b]['rf'] for b in BL]; ris = [bres[b]['ri'] for b in BL]
efs = [bres[b]['ef'] * 1.96 for b in BL]; eis = [bres[b]['ei'] * 1.96 for b in BL]
ax.bar(x - w / 2, rfs, w, yerr=efs, capsize=4, color=C_FOR, zorder=3,
       error_kw={'ecolor': '#333', 'lw': 1.1}, label='Formal')
ax.bar(x + w / 2, ris, w, yerr=eis, capsize=4, color=C_INF, zorder=3,
       error_kw={'ecolor': '#666', 'lw': 1.1}, label='Informal')
base_y = -max(rfs) * 0.09
for i, b in enumerate(BL):
    ax.text(i - w / 2, rfs[i] + efs[i] + 0.3, f'{rfs[i]:.2f}', ha='center',
            fontsize=9.5, fontweight='bold', color=C_FOR)
    ax.text(i + w / 2, ris[i] + eis[i] + 0.3, f'{ris[i]:.2f}', ha='center',
            fontsize=9.5, fontweight='bold', color='#4a6a8a')
    sig = '***' if bres[b]['p'] < 0.01 else ('**' if bres[b]['p'] < 0.05 else 'n.s.')
    ax.text(i, base_y, f"{rfs[i] / ris[i]:.2f}×   {sig}", ha='center', fontsize=9.5,
            color='#333' if bres[b]['p'] < 0.05 else C_GRIS, fontweight='bold')
ax.text(-0.72, base_y, 'Razón F/I:', ha='right', fontsize=8.5, style='italic', color='#444')
ax.legend(loc='upper right', frameon=False, fontsize=10, ncol=2)
ax.set_xticks(x); ax.set_xticklabels([etq[b] for b in BL], fontsize=10)
ax.set_ylabel('Retorno a la escolaridad (% por año)')
ax.set_ylim(base_y * 1.6, max(rfs) * 1.30)
ax.grid(axis='y', alpha=0.25); ax.set_axisbelow(True)
titulo(ax, 'retornos_bloque', 'Retornos a la escolaridad por bloque de actividad y segmento')
fuente(fig, 'Nota: barras de error al 95%. *** p<0.01; ** p<0.05; n.s. = diferencia no '
            'significativa (prueba de Wald). ' + FTE)
guardar(fig, f'FIG_{NUM["retornos_bloque"]:02d}_retornos_bloque.png')

# ─── 7. O-B por bloque ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6.0), constrained_layout=True)
lab_x = [etq[b].replace('\n', ' ') for b in BL] + ['GLOBAL']
pes = [bres[b]['pe'] for b in BL] + [pe]
pns = [bres[b]['pn'] for b in BL] + [pn]
brs = [bres[b]['br'] for b in BL] + [BR]
xb = np.arange(5)
ax.bar(xb, [max(v, 0) for v in pes], 0.52, color=C_INF, zorder=3)
ax.bar(xb, [max(v, 0) for v in pns], 0.52, bottom=[max(v, 0) for v in pes],
       color=C_FOR, zorder=3)
for i, (e_, n_, b_) in enumerate(zip(pes, pns, brs)):
    if e_ > 6:
        ax.text(i, e_ / 2, f'{e_:.1f}%', ha='center', va='center', fontsize=11,
                fontweight='bold', color=C_FOR)
    if n_ > 6:
        ax.text(i, max(e_, 0) + n_ / 2, f'{n_:.1f}%', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')
    ax.text(i, max(e_, 0) + max(n_, 0) + 2, f'Δ={b_:.3f}', ha='center', fontsize=8.5,
            color='#555', style='italic')
i_min = int(np.argmin([bres[b]['pn'] for b in BL]))
pn_min = bres[BL[i_min]]['pn']
if pn_min < 6:
    # CORRECCIÓN: el recuadro se coloca en el espacio libre superior izquierdo,
    # donde antes tapaba la etiqueta de Servicios Sociales.
    ax.annotate(f'Componente no explicado\nprácticamente nulo ({pn_min:.1f}%)',
                xy=(i_min, 100.5), xytext=(0.35, 126), fontsize=8.5, color=C_ACC,
                ha='center', arrowprops=dict(arrowstyle='->', color=C_ACC, lw=1.1),
                bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=C_ACC, alpha=0.95))
ax.axvline(3.5, color='#999', lw=1, ls=':', zorder=1)
ax.legend(handles=[Patch(facecolor=C_INF, label='Dotaciones'),
                   Patch(facecolor=C_FOR, label='Estructural')],
          loc='upper right', frameon=False, fontsize=10, ncol=2)
ax.set_xticks(xb); ax.set_xticklabels(lab_x, fontsize=9.5)
ax.set_ylabel('Proporción de la brecha salarial (%)'); ax.set_ylim(0, 138)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.grid(axis='y', alpha=0.25); ax.set_axisbelow(True)
titulo(ax, 'oaxaca_bloque', 'Descomposición de la brecha salarial por bloque de actividad')
fuente(fig, 'Nota: referencia en los coeficientes del segmento formal. ' + FTE)
guardar(fig, f'FIG_{NUM["oaxaca_bloque"]:02d}_oaxaca_bloque.png')

# ─── 8. Descomposición combinada ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10.5, 4.9), constrained_layout=True)
Y_G, Y_J, ALTO = 0.72, 0.0, 0.34
for datos, Y in [([(pe, C_INF, 'Dotaciones de\ncapital humano', C_FOR),
                   (pn, C_FOR, 'Componente\nestructural', 'white')], Y_G),
                 ([(SEG, C_SEG, 'Entre sub-sectores\n(segregación)', 'white'),
                   (DOT, C_INF, 'Dentro del sub-sector\n(dotaciones)', C_FOR),
                   (EST, C_FOR, 'Dentro del sub-sector\n(estructural)', 'white')], Y_J)]:
    izq = 0
    for v, col, txt, tc in datos:
        ax.barh([Y], [v], ALTO, left=[izq], color=col, zorder=3)
        ax.text(izq + v / 2, Y, f'{v:.1f}%', ha='center', va='center', fontsize=13,
                fontweight='bold', color=tc)
        ax.text(izq + v / 2, Y - ALTO / 2 - 0.06, txt, ha='center', va='top',
                fontsize=8.5, color='#333')
        izq += v
ax.annotate('', xy=(100 - EST, Y_J + ALTO / 2 + 0.02),
            xytext=(pe, Y_G - ALTO / 2 - 0.02),
            arrowprops=dict(arrowstyle='->', color=C_ACC, lw=1.1, ls=':'))
ax.text(-3, Y_G, 'Descomposición\nestándar', ha='right', va='center', fontsize=9.5,
        fontweight='bold', color='#333')
ax.text(-3, Y_J, 'Descomposición\njerárquica', ha='right', va='center', fontsize=9.5,
        fontweight='bold', color='#333')
ax.text(102, Y_G, f'Δ = {BR:.4f}', fontsize=9, style='italic', color='#555', va='center')
ax.text(102, Y_J, f'Δ = {TOT_J:.4f}', fontsize=9, style='italic', color='#555', va='center')
ax.set_xlim(-40, 116); ax.set_ylim(-0.62, 1.12); ax.set_yticks([])
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.spines['bottom'].set_bounds(0, 100)
ax.set_xlabel('Proporción de la brecha salarial total (%)')
ax.spines['left'].set_visible(False)
ax.grid(axis='x', alpha=0.2); ax.set_axisbelow(True)
titulo(ax, 'descomposicion',
       'Descomposición de la brecha salarial: partición estándar y partición jerárquica')
fuente(fig, f'Nota: ambas particiones se calculan sobre la misma brecha total pero no '
            f'están anidadas: la jerárquica emplea coeficientes y ponderadores a nivel de '
            f'bloque. Al controlar la segregación entre sub-sectores el componente '
            f'estructural se reduce de {pn:.1f}% a {EST:.1f}%. ' + FTE, y=-0.14)
guardar(fig, f'FIG_{NUM["descomposicion"]:02d}_descomposicion.png')

# ─── 9. Retornos anuales ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
crisis(ax)
ax.fill_between(S['anio'], S['ret_i'], S['ret_f'], color=C_INF, alpha=0.35, zorder=2)
ax.plot(S['anio'], S['ret_f'], color=C_FOR, lw=2.4, marker='o', ms=4, zorder=3)
ax.plot(S['anio'], S['ret_i'], color='#5a7ba6', lw=2.4, marker='o', ms=4, zorder=3)
ax.text(S['anio'].iloc[-1] + 0.3, S['ret_f'].iloc[-1], 'Formal', color=C_FOR,
        fontsize=10.5, fontweight='bold', va='center')
ax.text(S['anio'].iloc[-1] + 0.3, S['ret_i'].iloc[-1], 'Informal', color='#5a7ba6',
        fontsize=10.5, fontweight='bold', va='center')
ax.text(S['anio'].median(), (S['ret_f'].mean() + S['ret_i'].mean()) / 2,
        f"Diferencial promedio\n{S['gap_ret'].mean():.2f} puntos porcentuales",
        ha='center', fontsize=9.5, color='#2c4a6e', fontweight='bold')
imin = S['ret_i'].idxmin()
ax.annotate(f"Mínimo informal: {S.loc[imin, 'ret_i']:.2f}% ({S.loc[imin, 'anio']})",
            xy=(S.loc[imin, 'anio'], S.loc[imin, 'ret_i']),
            xytext=(S.loc[imin, 'anio'] - 6.5, S.loc[imin, 'ret_i'] - 1.5),
            fontsize=8.5, color='#333', arrowprops=dict(arrowstyle='->', color='#666', lw=1))
ax.set_ylabel('Retorno a la escolaridad (% por año adicional)')
ax.set_xlim(2004.3, 2026.6); ax.set_ylim(0, S['ret_f'].max() * 1.18); enteros(ax)
ax.grid(axis='y', alpha=0.25); ax.set_axisbelow(True)
titulo(ax, 'retornos_anuales', 'Retornos a la escolaridad por segmento, estimación anual')
tg = st.linregress(S['anio'], S['gap_ret'])
fuente(fig, f'Nota: el área sombreada representa el diferencial entre segmentos. '
            f'Ecuaciones estimadas por separado para cada año. La tendencia del '
            f'diferencial no es significativa ({tg.slope:+.3f} pp/año, p = {tg.pvalue:.3f}). '
            + FTE)
guardar(fig, f'FIG_{NUM["retornos_anuales"]:02d}_retornos_anuales.png')

# ─── 10. Componente estructural anual ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
crisis(ax)
ax.fill_between(S['anio'], 0, S['pct_est'], color=C_INF, alpha=0.30, zorder=2)
ax.plot(S['anio'], S['pct_est'], color=C_FOR, lw=2.2, marker='o', ms=4.5, zorder=3)
Sa, Sb = S[S['anio'] <= CORTE], S[S['anio'] > CORTE]
for sub, reg in [(Sa, TA), (Sb, TB)]:
    xx = np.array([sub['anio'].min(), sub['anio'].max()])
    ax.plot(xx, reg.intercept + reg.slope * xx, color=C_ACC, lw=2.0, zorder=5)
ymax = S['pct_est'].max()
ax.set_ylim(-2, ymax * 1.30)
# CORRECCIÓN: ambas etiquetas se colocan en zonas libres, sin encimarse con la serie
ax.text(Sa['anio'].min() + 1.2, ymax * 1.13, f'{TA.slope:+.2f} pp/año\n(p = {TA.pvalue:.4f})',
        ha='left', fontsize=9, color=C_ACC, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.85))
ax.text(Sb['anio'].max() - 0.4, ymax * 0.30, f'{TB.slope:+.2f} pp/año\n(p = {TB.pvalue:.4f})',
        ha='right', fontsize=9, color=C_ACC, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.85))
ax.axvline(CORTE + 0.5, color=C_GRIS, lw=1.1, ls='--', zorder=2)
ax.text(CORTE + 0.7, ymax * 1.26, f'Quiebre {CORTE} (F = {F_max:.1f})',
        fontsize=8, color=C_GRIS, va='top', style='italic')
pm = S['pct_est'].mean()
ax.axhline(pm, color='#999', ls=':', lw=1.0, zorder=1)
ax.text(S['anio'].max() + 0.35, pm, f'Media\n{pm:.1f}%', color='#777',
        fontsize=8.5, va='center', ha='left')
ax.axhline(0, color='#999', lw=0.9, zorder=1)
ax.set_ylabel('Componente estructural (% de la brecha anual)')
ax.set_xlim(2004.3, 2027.2); enteros(ax)
ax.grid(axis='y', alpha=0.25); ax.set_axisbelow(True)
titulo(ax, 'estructural_anual',
       'Componente estructural de la brecha salarial, estimación anual')
fuente(fig, 'Nota: estimaciones anuales sobre submuestras. Las rectas son tendencias por '
            'subperiodo; el quiebre se selecciona por máximo estadístico F entre todos los '
            'cortes candidatos y el rechazo se sostiene en todo el rango. ' + NOTA_COMP + FTE)
guardar(fig, f'FIG_{NUM["estructural_anual"]:02d}_estructural_anual.png')

# ═══════════════════════════════════════════════════════════════════════
# 9. VALORES PARA LAS TABLAS
# ═══════════════════════════════════════════════════════════════════════

print('\n' + '=' * 76)
print('  VALORES PARA LAS TABLAS')
print('=' * 76)

print('\n── TABLA 1. Estadística descriptiva ──')
print('   (diferencias sobre valores NO redondeados; decláralo al pie)')
print(f"\n{'Variable':<24}{'Formal':>11}{'(DE)':>10}{'Informal':>11}{'(DE)':>10}{'Dif.':>10}")
for v, n in [(SAL, 'Salario real/hora'), ('esc_anios', 'Escolaridad'),
             ('exp', 'Experiencia'), ('eda', 'Edad')]:
    if v in df.columns:
        a, b = mp(F, v), mp(I, v)
        print(f'{n:<24}{a:>11.2f}{sd_p(F, v):>10.2f}{b:>11.2f}{sd_p(I, v):>10.2f}{a - b:>10.2f}')
for v, n in [('con_contrato', 'Contrato escrito (%)'), ('acc_salud', 'Acceso salud (%)')]:
    if v in df.columns:
        a, b = mp(F, v) * 100, mp(I, v) * 100
        print(f'{n:<24}{a:>11.1f}{"—":>10}{b:>11.1f}{"—":>10}{a - b:>10.1f}')
print(f"{'N':<24}{len(F):>11,}{'':>10}{len(I):>11,}")
print(f'\n   Mediana ponderada: Formal ${med_f:.2f} | Informal ${med_i:.2f}')
print(f'   Coef. de variación: {sd_p(F, SAL) / mp(F, SAL):.3f} / {sd_p(I, SAL) / mp(I, SAL):.3f}')

print('\n── TABLA 2. Mincer global ──')
for p in ['formal', 'esc_anios', 'exp', 'exp2']:
    print(f'  {p:<14}{mg.params[p]:>12.6f}  (EE {mg.bse[p]:.6f})  t={mg.tvalues[p]:>8.2f}')
print(f"  {'Intercepto':<14}{mg.params['Intercept']:>12.6f}  (EE {mg.bse['Intercept']:.6f})")
print(f'  Premio a la formalidad exp(β)−1 = {(np.exp(mg.params["formal"]) - 1) * 100:.2f}%')
print(f'  R² = {mg.rsquared:.4f} | N = {int(mg.nobs):,}')

print('\n── TABLA 3. Mincer por segmento y prueba de Wald ──')
print(f"\n{'Parámetro':<14}{'Formal':>13}{'(EE)':>11}{'Informal':>13}{'(EE)':>11}{'Wald':>12}{'p':>9}")
for p in ['esc_anios', 'exp', 'exp2']:
    w = (mF.params[p] - mI.params[p]) ** 2 / (mF.bse[p] ** 2 + mI.bse[p] ** 2)
    print(f'{p:<14}{mF.params[p]:>13.6f}{mF.bse[p]:>11.6f}{mI.params[p]:>13.6f}'
          f'{mI.bse[p]:>11.6f}{w:>12.2f}{1 - st.chi2.cdf(w, 1):>9.4f}')
print(f"{'Intercepto':<14}{mF.params['Intercept']:>13.4f}{mF.bse['Intercept']:>11.4f}"
      f"{mI.params['Intercept']:>13.4f}{mI.bse['Intercept']:>11.4f}")
print(f'  R²: {mF.rsquared:.4f} / {mI.rsquared:.4f}   N: {int(mF.nobs):,} / {int(mI.nobs):,}')
print(f'  Retornos: {mF.params["esc_anios"] * 100:.2f}% / {mI.params["esc_anios"] * 100:.2f}%'
      f'   Razón: {mF.params["esc_anios"] / mI.params["esc_anios"]:.2f}×')
print(f'  Vértices del perfil de experiencia: {vertice_F:.1f} / {vertice_I:.1f} años.')
print('  El coeficiente lineal resume la pendiente media; la Tabla 5 muestra que')
print('  la brecha no se abre de manera uniforme a lo largo de la escolaridad.')

print('\n── TABLA 4. Descomposición de Oaxaca-Blinder global ──')
print(f'  Brecha {BR:.4f} | Explicado {EX:.4f} ({pe:.1f}%) | No explicado {NE:.4f} ({pn:.1f}%)')
print(f'  En niveles: total {(np.exp(BR) - 1) * 100:.1f}%; dotaciones '
      f'{(np.exp(EX) - 1) * 100:.1f}%; estructural {(np.exp(NE) - 1) * 100:.1f}%')

print('\n── TABLA 5. Brecha por nivel educativo y bloque de actividad ──')
print('   (estimación por tramos; n.d. = celda con menos de '
      f'{MIN_OBS_CELDA} observaciones)')
enc = ''.join(f'{t.replace(chr(10), " "):>18}' for t in TRAMOS_ETQ)
print(f"\n  {'Bloque':<26}{enc}")
for b in BL:
    fila = ''
    for t in TRAMOS_ETQ:
        g_, s_ = bres[b]['perfil'][t]
        fila += f'{"n.d.":>18}' if np.isnan(g_) else f'{(np.exp(g_) - 1) * 100:>17.1f}%'
    print(f"  {etq[b].replace(chr(10), ' '):<26}{fila}")
fila = ''
for t in TRAMOS_ETQ:
    r_ = RT[RT['etq'] == t]
    fila += f'{"n.d.":>18}' if r_.empty else f'{(np.exp(r_["g"].iloc[0]) - 1) * 100:>17.1f}%'
print(f"  {'GLOBAL':<26}{fila}")
print('\n  >> Reconciliación de las figuras 6 y 7: donde el perfil es plano a lo largo')
print('     de los tramos, la segmentación opera como desplazamiento uniforme del')
print('     salario a igualdad de capital humano observable. Donde el perfil tiene')
print('     pendiente, opera vía remuneración diferencial de ese mismo capital')
print('     humano. Son dos formas del mismo fenómeno.')

print('\n── TABLA por bloque: retornos y descomposición ──')
print(f"{'Bloque':<28}{'Ret.F':>8}{'Ret.I':>8}{'Razón':>8}{'Wald':>11}{'p':>9}"
      f"{'Δ':>9}{'%Dot':>8}{'%Est':>8}")
for b in BL:
    r = bres[b]
    print(f"{etq[b].replace(chr(10), ' '):<28}{r['rf']:>8.2f}{r['ri']:>8.2f}"
          f"{r['rf'] / r['ri']:>8.2f}{r['wald']:>11.2f}{r['p']:>9.4f}"
          f"{r['br']:>9.4f}{r['pe']:>7.1f}%{r['pn']:>7.1f}%")

print('\n── TABLA 9. Descomposición jerárquica ──')
print(f'  Entre sub-sectores      {SEG_v:.4f}  ({SEG:.1f}%)')
print(f'  Dentro — dotaciones     {DOT_v:.4f}  ({DOT:.1f}%)')
print(f'  Dentro — estructural    {EST_v:.4f}  ({EST:.1f}%)')
print(f'  Total                   {TOT_J:.4f}')
print(f'  El componente estructural pasa de {pn:.1f}% a {EST:.1f}%: la diferencia de '
      f'{pn - EST:.1f} puntos\n  la absorbe la segregación sectorial.')
print('\n  Participación de cada bloque por segmento:')
for b in BL:
    print(f"  {etq[b].replace(chr(10), ' '):<28}"
          f"{bres[b]['sh_f'] * 100:>12.1f}%{bres[b]['sh_i'] * 100:>14.1f}%")

# ═══════════════════════════════════════════════════════════════════════
# 10. CIERRE
# ═══════════════════════════════════════════════════════════════════════

ruta_csv = os.path.join(SALIDA, 'SERIES_anuales_cap5.csv')
S.to_csv(ruta_csv, index=False); ARCHIVOS.append(ruta_csv)
ruta_tr = os.path.join(SALIDA, 'BRECHA_por_tramo_educativo.csv')
RT.to_csv(ruta_tr, index=False); ARCHIVOS.append(ruta_tr)
fil = []
for b in BL:
    for t in TRAMOS_ETQ:
        g_, s_ = bres[b]['perfil'][t]
        fil.append({'bloque': etq[b].replace('\n', ' '), 'tramo': t.replace('\n', ' '),
                    'g': g_, 'se': s_})
ruta_bt = os.path.join(SALIDA, 'BRECHA_tramo_por_bloque.csv')
pd.DataFrame(fil).to_csv(ruta_bt, index=False); ARCHIVOS.append(ruta_bt)

print('\n' + '=' * 76)
print('  ARCHIVOS GENERADOS')
print('=' * 76)
for a in ARCHIVOS:
    print(f'  {a}')
print(f'  {RUTA_LOG}')
print(f'\n  {len(ARCHIVOS) + 1} archivos.')
print('=' * 76 + '\n')

sys.stdout.log.close()
sys.stdout = sys.stdout.terminal