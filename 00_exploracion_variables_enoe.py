"""
Exploración de variables educativas en los microdatos de la ENOE.

Compara la nomenclatura de las variables de escolaridad en cuatro trimestres
distribuidos a lo largo del periodo, para verificar que el esquema de
codificación se mantiene estable y determinar qué variable usar como medida
de años de instrucción.

Entrada  : microdatos ENOE, tabla SDEM (archivos trimestrales)
Salida   : reporte en consola
Uso      : diagnóstico previo. No modifica ni genera datos.

Tesis de licenciatura en Ingeniería Económica y Financiera
Luis Samuel Barria Gutiérrez — Universidad Autónoma del Estado de Hidalgo
"""

import pandas as pd

archivos = {
    "2005-T3": r"C:\Users\samue\OneDrive\Escritorio\inegi enoe microdatos\2005\SDEMT305.csv",
    "2010-T2": r"C:\Users\samue\OneDrive\Escritorio\inegi enoe microdatos\2010\SDEMT210.csv",
    "2016-T2": r"C:\Users\samue\OneDrive\Escritorio\inegi enoe microdatos\2016\SDEMT216.csv",
    "2025-T4": r"C:\Users\samue\OneDrive\Escritorio\inegi enoe microdatos\2025\ENOE_SDEMT425.csv",
}

for nombre, ruta in archivos.items():
    print("\n" + "="*60)
    print("ARCHIVO: " + nombre)
    print("="*60)

    df = pd.read_csv(ruta, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.lower().str.strip()

    vars_educ = [c for c in df.columns if any(x in c for x in
                 ['niv', 'esc', 'ins', 'edu', 'cs_p13', 'nivel'])]
    print("\n>>> Variables de educacion detectadas:")
    print(vars_educ)

    if 'niv_ins' in df.columns:
        print("\n>>> niv_ins - frecuencias:")
        print(df['niv_ins'].value_counts().sort_index())
    else:
        print("\n>>> niv_ins NO EXISTE")

    if 'cs_p13_1' in df.columns:
        print("\n>>> cs_p13_1 - frecuencias:")
        print(df['cs_p13_1'].value_counts().sort_index())

    if 'nivel_ins' in df.columns:
        print("\n>>> nivel_ins - frecuencias:")
        print(df['nivel_ins'].value_counts().sort_index())

    print("\n>>> ent existe: " + str('ent' in df.columns))
    print(">>> cve_ent existe: " + str('cve_ent' in df.columns))
    print(">>> Total filas: " + str(len(df)))   