"""
Auditoría de integridad de la base de análisis.

Verifica la consistencia interna de la base construida antes de estimar:
rango de la variable de escolaridad, estabilidad del sistema de clasificación
sectorial a lo largo del periodo, y trazabilidad del tamaño muestral.

Confirma que la exclusión del código 99 de la ENOE (escolaridad no
especificada) devuelve exactamente las 245,508 observaciones reportadas en el
documento.

Entrada  : BASE_FINAL_CORREGIDA_2025.csv
Salida   : reporte en consola

Tesis de licenciatura en Ingeniería Económica y Financiera
Luis Samuel Barria Gutiérrez — Universidad Autónoma del Estado de Hidalgo
"""

import pandas as pd, numpy as np
df = pd.read_csv(r"C:\Users\REGLAMENTOS\Desktop\BASE_FINAL_CORREGIDA_2025.csv")

print("=== 1. ESCOLARIDAD ===")
print(f"Máximo: {df['esc_anios'].max()}")
print(f"Casos con 99: {(df['esc_anios']==99).sum():,}")
print(f"Casos >24:    {(df['esc_anios']>24).sum():,}")

print("\n=== 2. RUBRO ===")
print(f"Valores únicos: {sorted(df['rubro'].dropna().unique())}")
print(f"Fuera de 6-19:  {(~df['rubro'].between(6,19)).sum():,}")
print("\nRubros por año (¿cambia el sistema?):")
print(df.groupby('anio')['rubro'].agg(['min','max']))

print("\n=== 3. INTEGRIDAD GENERAL ===")
print(f"N total: {len(df):,}")
print(df[['esc_anios','exp','ing_hr_real_wins']].describe().round(2))

# ¿Filtrar esc_anios==99 nos regresa a 245,508?
df_filtrado = df[df['esc_anios'] != 99].copy()
print(f"N después de quitar esc_anios=99: {len(df_filtrado):,}")
print(f"¿Coincide con 245,508? {len(df_filtrado) == 245508}")

# Si coincide, verificamos que las medias del documento salgan de aquí
print("\n=== Medias por segmento (filtrado) ===")
print(df_filtrado.groupby('formal')['esc_anios'].mean().round(2))
print(df_filtrado.groupby('formal')['exp'].mean().round(2))
print(df_filtrado.groupby('formal').size())

print("\n=== Medias por segmento (SIN filtrar, para comparar) ===")
print(df.groupby('formal')['esc_anios'].mean().round(2))