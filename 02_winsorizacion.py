"""
Winsorización del ingreso real por hora.

Aplica el tratamiento de valores extremos en los percentiles 1 y 99 sobre el
ingreso real por hora y genera la variable dependiente en logaritmos utilizada
en todas las estimaciones.

Entrada  : BASE_FINAL_CORREGIDA_2025.csv
Salida   : BASE_WINSORIZADA_2025.csv

Tesis de licenciatura en Ingeniería Económica y Financiera
Luis Samuel Barria Gutiérrez — Universidad Autónoma del Estado de Hidalgo
"""

import pandas as pd
import numpy as np
import os

# ── RUTAS ────────────────────────────────────────────────────────────────────
RUTA = r"C:\Users\REGLAMENTOS\Downloads"
ENTRADA = os.path.join(RUTA, "BASE_FINAL_CORREGIDA_2025.csv")
INPC_FILE = os.path.join(RUTA, "INPC_TRIMESTRAL_2005_2025.csv")
SALIDA  = os.path.join(RUTA, "BASE_WINSORIZADA_2025.csv")

print("="*70)
print("  PASO 3: WINSORIZACIÓN — SALARIO REAL POR HORA")
print("="*70)

# ── CARGA ─────────────────────────────────────────────────────────────────────
if not os.path.exists(ENTRADA):
    raise FileNotFoundError(f"No se encontró: {ENTRADA}")

df = pd.read_csv(ENTRADA, low_memory=False)
print(f"\n✅ Base cargada: {len(df):,} observaciones")
print(f"   Columnas: {df.columns.tolist()}")

# ── VERIFICAR / CONSTRUIR ing_hr_real ────────────────────────────────────────
if 'ing_hr_real' not in df.columns:
    print("\n⚠️  'ing_hr_real' no existe. Calculando desde ing_hr + INPC...")
    if not os.path.exists(INPC_FILE):
        raise FileNotFoundError(f"Necesito el INPC en: {INPC_FILE}")
    inpc = pd.read_csv(INPC_FILE).dropna(subset=['anio','trimestre','inpc'])
    inpc['anio']      = inpc['anio'].astype(int)
    inpc['trimestre'] = inpc['trimestre'].astype(int)
    pivote_2018 = inpc[inpc['anio'] == 2018]['inpc'].mean()
    print(f"   Pivote 2018: {pivote_2018:.4f}")
    df = df.merge(inpc[['anio','trimestre','inpc']], on=['anio','trimestre'], how='left')
    df['ing_hr_real'] = df['ing_hr'] * (pivote_2018 / df['inpc'])
    print(f"   ing_hr_real creado. NaN: {df['ing_hr_real'].isna().sum():,}")
else:
    print("✅ 'ing_hr_real' encontrada en la base.")

# ── FILTRO: solo ingresos positivos ──────────────────────────────────────────
n_antes = len(df)
df = df[df['ing_hr_real'] > 0].copy()
print(f"   Con ingreso > 0: {len(df):,}  (eliminados: {n_antes - len(df):,})")

# ── WINSORIZACIÓN ─────────────────────────────────────────────────────────────
p1  = df['ing_hr_real'].quantile(0.01)
p99 = df['ing_hr_real'].quantile(0.99)

n_bajo = (df['ing_hr_real'] < p1).sum()
n_alto = (df['ing_hr_real'] > p99).sum()
n_afectadas = n_bajo + n_alto
pct = n_afectadas / len(df) * 100

print()
print("─"*60)
print("  *** NÚMEROS PARA EL CAPÍTULO 4 DE TU TESIS ***")
print("─"*60)
print(f"  Límite inferior (p1):    ${p1:.2f} pesos reales/hora")
print(f"  Límite superior (p99):   ${p99:.2f} pesos reales/hora")
print(f"  Obs. por debajo del p1:  {n_bajo:,}")
print(f"  Obs. por encima del p99: {n_alto:,}")
print(f"  Total afectadas:         {n_afectadas:,}  ({pct:.1f}% de la muestra)")
print("─"*60)

# Aplicar winsorización
df['ing_hr_real_wins'] = df['ing_hr_real'].clip(lower=p1, upper=p99)
df['ln_ing_hr_real_wins'] = np.log(df['ing_hr_real_wins'])

# ── ESTADÍSTICAS POR SEGMENTO ────────────────────────────────────────────────
if 'formal' in df.columns:
    print()
    print("  BRECHAS SALARIALES (confirmar vs documento):")
    yf = df[df['formal']==1]['ln_ing_hr_real_wins'].mean()
    yi = df[df['formal']==0]['ln_ing_hr_real_wins'].mean()
    brecha_ln  = yf - yi
    brecha_pct = (np.exp(brecha_ln) - 1) * 100
    print(f"  ln salario formal:   {yf:.4f}")
    print(f"  ln salario informal: {yi:.4f}")
    print(f"  Brecha (ln):         {brecha_ln:.4f}")
    print(f"  Brecha en niveles:   {brecha_pct:.1f}%")
    print(f"  (Documento decía: 54.7% — ¿coincide?)")

# ── VERIFICACIÓN MÍNIMA ───────────────────────────────────────────────────────
assert df['ln_ing_hr_real_wins'].isna().sum() == 0
assert np.isfinite(df['ln_ing_hr_real_wins']).all()
print("\n✅ Sin NaN ni valores infinitos en ln_ing_hr_real_wins")

# ── GUARDAR ───────────────────────────────────────────────────────────────────
df.to_csv(SALIDA, index=False)
print(f"\n✅ Guardado: BASE_WINSORIZADA_2025.csv")
print(f"   Observaciones: {len(df):,}")
print(f"   Variable lista: ln_ing_hr_real_wins")
print()
print("="*70)