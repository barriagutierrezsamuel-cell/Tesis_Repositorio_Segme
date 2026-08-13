"""
Pruebas de robustez del Anexo C.

Reproduce los tres ejercicios de robustez bajo la especificación oficial, de
modo que las tres comparaciones partan de la misma base metodológica que el
capítulo de resultados:

  C.1  Tratamiento de valores extremos
       ingreso winsorizado frente a ingreso sin winsorizar
  C.2  Convención de grupo de referencia
       coeficientes del segmento formal frente a los del informal
  C.3  Definición de la población objetivo
       subordinados y remunerados frente a solo asalariados

El procedimiento emite una advertencia si alguna cifra difiere de la
registrada en el documento.

Entrada  : BASE_FINAL_CORREGIDA_2025.csv
Salida   : reporte en consola

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

RUTA   = r"C:\Users\REGLAMENTOS\Desktop\BASE_FINAL_CORREGIDA_2025.csv"
SALIDA = r"C:\Users\REGLAMENTOS\Desktop\FIGURAS_TESIS"

HK = ['esc_anios', 'exp', 'exp2']
FIJOS = "C(anio, Treatment(reference=2005))"

# Cifras del Capítulo 5, para contraste
OFICIAL = {'brecha': 0.4297, 'dot_pct': 82.7, 'est_pct': 17.3}
# Cifras registradas en el documento para C.1 y C.2 con la especificación
# anterior; el script verifica si siguen siendo válidas
PREVIO_C1 = 81.1   # dotaciones, muestra sin winsorizar
PREVIO_C2 = 33.6   # dotaciones, referencia informal


class Tee:
    def __init__(self, ruta):
        self.terminal, self.log = sys.stdout, open(ruta, 'w', encoding='utf-8')

    def write(self, m):
        self.terminal.write(m); self.log.write(m)

    def flush(self):
        self.terminal.flush(); self.log.flush()


os.makedirs(SALIDA, exist_ok=True)
sys.stdout = Tee(os.path.join(SALIDA, 'ANEXO_C_robustez.txt'))

print('\n' + '=' * 78)
print('  ANEXO C — PRUEBAS DE ROBUSTEZ')
print('  Especificación oficial: WLS ponderado, HC3, efectos fijos de año')
print('=' * 78)

if not os.path.exists(RUTA):
    print(f'\n  ✗ ERROR: no se encuentra {RUTA}'); sys.exit(1)

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

DISPONIBLES = [c for c in ['ln_ing_hr_real_wins', 'ln_ing_hr_real']
               if c in df.columns]
print(f'   Variables dependientes disponibles: {DISPONIBLES}')
if 'ln_ing_hr_real' not in df.columns:
    print('   ⚠ No existe ln_ing_hr_real: el ejercicio C.1 no puede realizarse')
    print('     con esta base. Habría que reconstruirla conservando la serie')
    print('     sin winsorizar.')


def mp(d, c):
    return np.average(d[c], weights=d['factor'])


def descomponer(datos, vd, referencia='formal'):
    """Descomposición de Oaxaca-Blinder en dos componentes.
    referencia: 'formal' usa β_F; 'informal' usa β_I."""
    d = datos[datos[vd].notna()].copy()
    F, I = d[d['formal'] == 1], d[d['formal'] == 0]
    form = f"{vd} ~ esc_anios + exp + exp2 + {FIJOS}"
    mF = smf.wls(form, data=F, weights=F['factor']).fit(cov_type='HC3')
    mI = smf.wls(form, data=I, weights=I['factor']).fit(cov_type='HC3')
    br = mp(F, vd) - mp(I, vd)
    beta = mF.params if referencia == 'formal' else mI.params
    dot = sum((mp(F, v) - mp(I, v)) * beta[v] for v in HK)
    return {
        'n': len(d), 'br': br, 'dot': dot, 'est': br - dot,
        'pd': dot / br * 100, 'pe_': (br - dot) / br * 100,
        'rf': mF.params['esc_anios'] * 100, 'ri': mI.params['esc_anios'] * 100,
    }


def linea(nom, r):
    print(f"  {nom:<40}{r['n']:>10,}{r['br']:>9.4f}{r['rf']:>8.2f}"
          f"{r['ri']:>8.2f}{r['rf'] / r['ri']:>7.2f}{r['pd']:>8.1f}%{r['pe_']:>8.1f}%")


ENC = (f"  {'Especificación':<40}{'N':>10}{'Brecha':>9}{'Ret.F':>8}"
       f"{'Ret.I':>8}{'Razón':>7}{'Dotac.':>9}{'Estruc.':>9}")

# ═══════════════════════════════════════════════════════════════════════
# REFERENCIA: especificación principal
# ═══════════════════════════════════════════════════════════════════════

print('\n' + '─' * 78)
print('  ESPECIFICACIÓN PRINCIPAL (Capítulo 5)')
print('─' * 78)
print(ENC)
base = descomponer(df, 'ln_ing_hr_real_wins', 'formal')
linea('Winsorizada, referencia formal', base)

ok = (abs(base['br'] - OFICIAL['brecha']) < 0.0025 and
      abs(base['pd'] - OFICIAL['dot_pct']) < 0.3)
print(f"\n  {'✓' if ok else '✗'} Contraste con el Capítulo 5: "
      f"brecha {OFICIAL['brecha']:.4f}, dotaciones {OFICIAL['dot_pct']:.1f}%")
if not ok:
    print('  ✗ DETENIDO: la especificación principal no se reproduce.')
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════
# C.1  WINSORIZACIÓN
# ═══════════════════════════════════════════════════════════════════════

print('\n' + '─' * 78)
print('  C.1  TRATAMIENTO DE VALORES EXTREMOS')
print('─' * 78)
if 'ln_ing_hr_real' in df.columns:
    print(ENC)
    linea('Con winsorización (principal)', base)
    sinw = descomponer(df, 'ln_ing_hr_real', 'formal')
    linea('Sin winsorización', sinw)
    print(f'\n  Diferencia en el componente de dotaciones: '
          f'{sinw["pd"] - base["pd"]:+.1f} puntos porcentuales.')
    if abs(sinw['pd'] - PREVIO_C1) > 0.5:
        print(f'  ⚠ El documento registra {PREVIO_C1:.1f}% para esta fila, obtenido')
        print(f'    con la especificación anterior. El valor vigente es '
              f'{sinw["pd"]:.1f}%.')
        print('    Actualiza la cifra en el Anexo C y en cualquier mención del texto.')
    else:
        print(f'  ✓ Coincide con la cifra registrada ({PREVIO_C1:.1f}%).')
else:
    print('  No realizable: la base no conserva la variable sin winsorizar.')

# ═══════════════════════════════════════════════════════════════════════
# C.2  CONVENCIÓN DE REFERENCIA
# ═══════════════════════════════════════════════════════════════════════

print('\n' + '─' * 78)
print('  C.2  CONVENCIÓN DE GRUPO DE REFERENCIA')
print('─' * 78)
print(ENC)
linea('Referencia formal (principal)', base)
ref_i = descomponer(df, 'ln_ing_hr_real_wins', 'informal')
linea('Referencia informal', ref_i)
print(f'\n  El componente de dotaciones pasa de {base["pd"]:.1f}% a {ref_i["pd"]:.1f}%')
print(f'  y el estructural de {base["pe_"]:.1f}% a {ref_i["pe_"]:.1f}%.')
if abs(ref_i['pd'] - PREVIO_C2) > 0.5:
    print(f'  ⚠ El documento registra {PREVIO_C2:.1f}% / {100 - PREVIO_C2:.1f}% en la')
    print(f'    Sección 5.5 y en el Anexo C. El valor vigente es '
          f'{ref_i["pd"]:.1f}% / {ref_i["pe_"]:.1f}%.')
    print('    Corrige ambas menciones.')
else:
    print(f'  ✓ Coincide con la cifra registrada ({PREVIO_C2:.1f}%).')

# ═══════════════════════════════════════════════════════════════════════
# C.3  DEFINICIÓN DE LA POBLACIÓN OBJETIVO
# ═══════════════════════════════════════════════════════════════════════

print('\n' + '─' * 78)
print('  C.3  DEFINICIÓN DE LA POBLACIÓN OBJETIVO')
print('─' * 78)
if 'remune2c' in df.columns:
    print(ENC)
    linea('Subordinados y remunerados (principal)', base)
    asal = descomponer(df[df['remune2c'] == 1], 'ln_ing_hr_real_wins', 'formal')
    linea('Solo asalariados', asal)
    print(f'\n  La restricción {"eleva" if asal["br"] > base["br"] else "reduce"} '
          f'la brecha en {abs(asal["br"] - base["br"]):.4f} puntos logarítmicos')
    print(f'  y {"eleva" if asal["rf"] / asal["ri"] > base["rf"] / base["ri"] else "reduce"} '
          f'la razón de retornos de {base["rf"] / base["ri"]:.2f} a '
          f'{asal["rf"] / asal["ri"]:.2f}.')
    print('\n  Composición de la muestra:')
    w = df.groupby('remune2c')['factor'].sum()
    for k, lab in [(1, 'Asalariados'), (2, 'Con percepciones no salariales')]:
        s = df[df['remune2c'] == k]
        pf = s[s['formal'] == 1]['factor'].sum() / df[df['formal'] == 1]['factor'].sum()
        pi = s[s['formal'] == 0]['factor'].sum() / df[df['formal'] == 0]['factor'].sum()
        print(f'    {lab:<32}{w[k] / w.sum() * 100:>7.1f}% del total   '
              f'{pf * 100:>6.1f}% del empleo formal   {pi * 100:>6.1f}% del informal')
else:
    print('  No realizable: la base no contiene la variable remune2c.')

# ═══════════════════════════════════════════════════════════════════════
# RESUMEN
# ═══════════════════════════════════════════════════════════════════════

print('\n' + '=' * 78)
print('  RESUMEN PARA LA TABLA DEL ANEXO C')
print('=' * 78)
print(ENC)
linea('Especificación principal', base)
if 'ln_ing_hr_real' in df.columns:
    linea('C.1  Sin winsorización', sinw)
linea('C.2  Referencia informal', ref_i)
if 'remune2c' in df.columns:
    linea('C.3  Solo asalariados', asal)
print('\n  Todas las filas emplean la misma especificación de Mincer, la misma')
print('  ponderación y los mismos efectos fijos de año. Solo varía el elemento')
print('  que cada ejercicio somete a prueba.')
print('=' * 78 + '\n')

sys.stdout.log.close()
sys.stdout = sys.stdout.terminal