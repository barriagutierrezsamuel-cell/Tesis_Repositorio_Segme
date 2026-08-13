"""
Construcción de la base de análisis.

Apila los 83 trimestres de la ENOE correspondientes al periodo 2005-2025,
restringe la muestra a trabajadores subordinados y remunerados del sector
terciario privado de Ciudad de México y Estado de México, construye las
variables de capital humano y deflacta los ingresos nominales con el INPC.

Especificaciones:
  - Escolaridad tomada de la variable anios_esc de la ENOE
  - Experiencia potencial = edad - años de escolaridad - 6, truncada en cero
  - Deflactación con INPC trimestral, base 2018 = 100
  - Winsorización P1-P99 del ingreso real por hora

Entrada  : microdatos ENOE (SDEM trimestral) e INPC trimestral
Salida   : BASE_FINAL_CORREGIDA_2025.csv

Tesis de licenciatura en Ingeniería Económica y Financiera
Luis Samuel Barria Gutiérrez — Universidad Autónoma del Estado de Hidalgo
"""

import pandas as pd
import numpy as np
import os
import glob
import re


# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
RUTA_BASE  = r"C:\Users\samue\OneDrive\Escritorio\inegi enoe microdatos"
RUTA_INPC  = r"C:\Users\samue\OneDrive\Escritorio\inegi enoe microdatos\INPC_TRIMESTRAL_2005_2025.csv"
RUTA_SALIDA = os.path.join(RUTA_BASE, "BASE_FINAL_CORREGIDA_2025.csv")

# INPC base 2018 = 100
# El CSV debe tener columnas: anio, trimestre, inpc
# Si no tienes el archivo, el script lo avisa y continúa sin deflactar

# =============================================================================
lista_df = []

print("\n" + "="*70)
print("  PIPELINE MAESTRO CORREGIDO — ENOE SUBORDINADOS 2005-2025")
print("="*70 + "\n")

# ── CARGA INPC ────────────────────────────────────────────────────────────────
try:
    # El CSV tiene columnas basura a la izquierda y las columnas limpias
    # (anio, trimestre, inpc) a la derecha — se leen directamente por nombre
    _raw     = pd.read_csv(RUTA_INPC, low_memory=False)
    inpc_df  = _raw[['anio', 'trimestre', 'inpc']].dropna().copy()
    inpc_df['anio']      = inpc_df['anio'].astype(int)
    inpc_df['trimestre'] = inpc_df['trimestre'].astype(int)
    inpc_df['inpc']      = pd.to_numeric(inpc_df['inpc'], errors='coerce')
    inpc_df  = inpc_df.dropna(subset=['inpc']).reset_index(drop=True)
    print(f"✅ INPC cargado: {len(inpc_df)} trimestres ({inpc_df['anio'].min()}–{inpc_df['anio'].max()})\n")
except Exception as e:
    inpc_df = None
    print(f"⚠️  INPC no encontrado ({e}). Se omitirá deflactación.\n")

# ── LOOP PRINCIPAL ────────────────────────────────────────────────────────────
for anio_folder in sorted(os.listdir(RUTA_BASE)):
    if not anio_folder.isdigit():
        continue
    anio = int(anio_folder)
    path_anio = os.path.join(RUTA_BASE, anio_folder)

    # Captura SDEMT y ENOE_SDEMT (cubre 2025-T4: ENOE_SDEMT425.csv)
    archivos = glob.glob(os.path.join(path_anio, "*[sS][dD][eE][mM][tT]*.csv"))

    for archivo in sorted(archivos):
        nombre = os.path.basename(archivo).lower()

        try:
            # ── 1. CARGA ──────────────────────────────────────────────────────
            df = pd.read_csv(archivo, encoding="latin-1", low_memory=False)
            df.columns = df.columns.str.lower().str.strip().str.replace('[^a-z0-9_]', '', regex=True)

            # ── 2. TRIMESTRE ──────────────────────────────────────────────────
            m = re.search(r'sdemt\s*([1-4])\d{2}', nombre)
            if not m:
                print(f"  ⚠️  No se detectó trimestre en: {nombre}")
                continue
            trimestre = int(m.group(1))

            # ── 3. AJUSTE GEOGRÁFICO (candado 2025-T3+) ───────────────────────
            if 'ent' not in df.columns and 'cve_ent' in df.columns:
                df['ent'] = df['cve_ent']
            if 'ent' not in df.columns:
                print(f"  ⚠️  Sin variable geográfica: {nombre}")
                continue

            # ── 4. FILTRO: CDMX(9) + EDOMEX(15) + SUBORDINADOS (pos_ocu=1) ───
            if 'pos_ocu' not in df.columns:
                continue
            df = df[(df['ent'].isin([9, 15])) & (df['pos_ocu'] == 1)].copy()
            if df.empty:
                continue

            # ── 5. ARMONIZACIÓN SECTORIAL ─────────────────────────────────────
            df['scian_num'] = pd.to_numeric(df.get('scian', np.nan), errors='coerce')

            if df['scian_num'].max(skipna=True) <= 21:
                # Sistema 21 sectores (2005-2012 aprox.)
                df = df[df['scian_num'].between(6, 19)].copy()
                df['rubro'] = df['scian_num']
            else:
                # SCIAN 2 dígitos (2013+)
                df['scian_2d'] = df['scian'].astype(str).str.replace(r'\.0$', '', regex=True).str[:2]
                df['scian_2d'] = pd.to_numeric(df['scian_2d'], errors='coerce')
                df = df[df['scian_2d'].between(43, 81) & (df['scian_2d'] != 93)].copy()
                df['rubro'] = df['scian_2d']

            if df.empty:
                continue

            # ── 6. FORMALIDAD (emp_ppal: 2=Formal, 1=Informal) ───────────────
            if 'emp_ppal' not in df.columns:
                continue
            df['formal_cat'] = pd.to_numeric(df['emp_ppal'], errors='coerce')
            df['formal']     = (df['formal_cat'] == 2).astype(int)

            # ── 7. EDAD ───────────────────────────────────────────────────────
            df['eda'] = pd.to_numeric(df['eda'], errors='coerce')
            df = df[df['eda'].between(15, 80)].copy()
            df['eda2'] = df['eda'] ** 2

            # ── 8. EDUCACIÓN — CORRECCIÓN [1] Y [2] ──────────────────────────
            # [1] Mapeo niv_ins CORRECTO (código 0 = Sin instrucción)
            df['niv_ins_num'] = pd.to_numeric(df.get('niv_ins', np.nan), errors='coerce')
            df['educ_nivel']  = df['niv_ins_num'].map({
                0: 'Sin_Inst',
                1: 'Primaria',
                2: 'Secundaria',
                3: 'Media_Sup',
                4: 'Superior',
                5: 'Posgrado'
            }).fillna('Otras')

            # [2] anios_esc: usar variable ENOE directamente
            if 'anios_esc' in df.columns:
                df['esc_anios'] = pd.to_numeric(df['anios_esc'], errors='coerce')
            else:
                # Fallback: proxy por nivel (solo si anios_esc no existe)
                df['esc_anios'] = df['niv_ins_num'].map({
                    0:  0,
                    1:  6,
                    2:  9,
                    3: 12,
                    4: 16,
                    5: 18
                })

            # [3] Experiencia potencial Mincer con anios_esc real
            df['exp']  = (df['eda'] - df['esc_anios'] - 6).clip(lower=0)
            df['exp2'] = df['exp'] ** 2

            # ── 9. INGRESO NOMINAL POR HORA ───────────────────────────────────
            df['ing_hr'] = pd.to_numeric(
                df.get('ing_x_hrs2', df.get('ing_x_hrs', np.nan)), errors='coerce'
            )
            # Fallback: ingreso ocupación / horas
            mask_sin_inghr = df['ing_hr'].isna() | (df['ing_hr'] <= 0)
            if mask_sin_inghr.any() and 'ingocup' in df.columns and 'hrsocup' in df.columns:
                hrs = pd.to_numeric(df['hrsocup'], errors='coerce').replace(0, np.nan)
                ing = pd.to_numeric(df['ingocup'], errors='coerce')
                df.loc[mask_sin_inghr, 'ing_hr'] = ing / hrs

            # Filtro: solo ingreso positivo
            df = df[df['ing_hr'] > 0].copy()
            if df.empty:
                continue

            # ── 10. VARIABLES DE CALIDAD LABORAL ──────────────────────────────
            for v in ['mh_col', 'pre_asa', 'tip_con', 'remune2c']:
                df[v] = pd.to_numeric(df.get(v, np.nan), errors='coerce')

            # ── 11. LOCALIDAD ─────────────────────────────────────────────────
            if 't_loc_tri' in df.columns:
                df['t_loc']        = pd.to_numeric(df['t_loc_tri'], errors='coerce')
                df['t_loc_fuente'] = 't_loc_tri (2020-2025)'
            elif 't_loc' in df.columns:
                df['t_loc']        = pd.to_numeric(df['t_loc'], errors='coerce')
                df['t_loc_fuente'] = 't_loc (2005-2019)'
            else:
                df['t_loc'], df['t_loc_fuente'] = np.nan, 'No detectada'

            # ── 12. FACTOR DE EXPANSIÓN ───────────────────────────────────────
            df['factor'] = pd.to_numeric(
                df.get('fac_tri', df.get('fac', 1)), errors='coerce'
            ).fillna(1)

            # ── 13. METADATOS ─────────────────────────────────────────────────
            df['anio']      = anio
            df['trimestre'] = trimestre

            # ── 14. SELECCIÓN FINAL ───────────────────────────────────────────
            cols = [
                'anio', 'trimestre', 'ent', 'sex',
                'eda', 'eda2', 'esc_anios', 'educ_nivel',
                'exp', 'exp2',
                'rubro', 'formal', 'formal_cat',
                'ing_hr', 'factor',
                'mh_col', 'pre_asa', 'tip_con', 'remune2c',
                't_loc', 't_loc_fuente'
            ]
            df_out = df[[c for c in cols if c in df.columns]].copy()
            lista_df.append(df_out)

            inf  = (df_out['formal'] == 0).sum()
            for_ = (df_out['formal'] == 1).sum()
            print(f"  ✅ {nombre:<30} | n={len(df_out):>6,} | F:{for_:>5,} / I:{inf:>5,}")

        except Exception as e:
            print(f"  ❌ Error en {nombre}: {e}")

# =============================================================================
# CONSOLIDACIÓN
# =============================================================================
if not lista_df:
    print("\n❌ No se procesó ningún archivo. Revisa la ruta base.")
else:
    print("\n" + "="*70)
    print("  CONSOLIDANDO BASE FINAL...")
    print("="*70)

    base = pd.concat(lista_df, ignore_index=True)

    # ── DEFLACTACIÓN CON INPC ─────────────────────────────────────────────────
    if inpc_df is not None:
        base = base.merge(inpc_df[['anio', 'trimestre', 'inpc']],
                          on=['anio', 'trimestre'], how='left')
        # INPC base 2018 = 100 → salario real en pesos de 2018
        # Base 2018 = promedio anual de los 4 trimestres de 2018
        _2018     = inpc_df[inpc_df['anio'] == 2018]['inpc']
        base_2018 = _2018.mean() if len(_2018) > 0 else 100.0
        base['ing_hr_real'] = base['ing_hr'] * (base_2018 / base['inpc'])
    else:
        base['ing_hr_real'] = base['ing_hr']
        base['inpc']        = np.nan

    # ── WINSORIZACIÓN P1-P99 ──────────────────────────────────────────────────
    p1  = base['ing_hr_real'].quantile(0.01)
    p99 = base['ing_hr_real'].quantile(0.99)
    base['ing_hr_real_wins'] = base['ing_hr_real'].clip(lower=p1, upper=p99)
    print(f"\n  Winsorización ing_hr_real: P1={p1:.2f} | P99={p99:.2f}")
    print(f"  Obs winzorizadas: {((base['ing_hr_real'] < p1) | (base['ing_hr_real'] > p99)).sum():,}")

    # ── LOG-SALARIO ───────────────────────────────────────────────────────────
    base['ln_ing_hr_real']      = np.log(base['ing_hr_real'].replace(0, np.nan))
    base['ln_ing_hr_real_wins'] = np.log(base['ing_hr_real_wins'].replace(0, np.nan))

    # ── EXCLUIR 2020-T2 (ETOE — no comparable) ───────────────────────────────
    n_antes = len(base)
    base = base[~((base['anio'] == 2020) & (base['trimestre'] == 2))].copy()
    print(f"  2020-T2 excluido: {n_antes - len(base):,} obs removidas (ETOE)")

    # ── GUARDAR ───────────────────────────────────────────────────────────────
    base.to_csv(RUTA_SALIDA, index=False)

    # ── REPORTE FINAL ─────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  REPORTE FINAL")
    print("="*70)
    print(f"  Observaciones totales : {len(base):,}")
    print(f"  Variables             : {base.shape[1]}")
    print(f"  Periodo               : {base['anio'].min()} – {base['anio'].max()}")
    print(f"  Formales              : {(base['formal']==1).sum():,} ({(base['formal']==1).mean()*100:.1f}%)")
    print(f"  Informales            : {(base['formal']==0).sum():,} ({(base['formal']==0).mean()*100:.1f}%)")
    print(f"\n  Distribución educ_nivel:")
    print(base['educ_nivel'].value_counts().to_string())
    print(f"\n  esc_anios describe:")
    print(base['esc_anios'].describe().round(2).to_string())
    print(f"\n  exp describe:")
    print(base['exp'].describe().round(2).to_string())
    print(f"\n✔  BASE GUARDADA EN: {RUTA_SALIDA}")




    # Diagnóstico: ¿cuántos tienen niv_ins=0 antes y después del filtro sectorial?
print("\nDiagnóstico Sin_Inst:")
print(base[base['educ_nivel']=='Otras'].groupby('anio').size())