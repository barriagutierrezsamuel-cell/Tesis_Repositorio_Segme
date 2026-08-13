"""
Figura de la descomposición jerárquica de la brecha salarial.

Representa los tres componentes de la partición jerárquica producidos por el
script 07 como una barra apilada que suma el cien por ciento de la brecha.

Entrada  : valores del script 07_descomposicion_jerarquica.py
Salida   : figura en formato PNG

Tesis de licenciatura en Ingeniería Económica y Financiera
Luis Samuel Barria Gutiérrez — Universidad Autónoma del Estado de Hidalgo
"""

import matplotlib.pyplot as plt
import numpy as np


# Componentes (% de la brecha total 0.4297)
SEGREGACION = 47.7   # entre bloques
DOT_WITHIN  = 39.1   # dentro – dotaciones
EST_WITHIN  = 13.3   # dentro – estructural
BRECHA      = 0.4297

COLOR_SEG = '#c8956f'   # segregación (tono cálido, la distingue del O-B)
COLOR_DOT = '#8fa8c8'   # dotaciones (mismo que O-B)
COLOR_EST = '#1a2e4a'   # estructural (mismo que O-B)

fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)

x = 0
w = 0.5
b1 = ax.bar(x, SEGREGACION, w, color=COLOR_SEG, zorder=3,
            label='Entre sub-sectores\n(segregación / composición)')
b2 = ax.bar(x, DOT_WITHIN, w, bottom=SEGREGACION, color=COLOR_DOT, zorder=3,
            label='Dentro del sub-sector\n(dotaciones de capital humano)')
b3 = ax.bar(x, EST_WITHIN, w, bottom=SEGREGACION+DOT_WITHIN, color=COLOR_EST,
            zorder=3, label='Dentro del sub-sector\n(estructural / segmentación)')

# Etiquetas dentro de cada segmento
ax.text(x, SEGREGACION/2, f'{SEGREGACION:.1f}%', ha='center', va='center',
        fontsize=13, fontweight='bold', color='white')
ax.text(x, SEGREGACION + DOT_WITHIN/2, f'{DOT_WITHIN:.1f}%', ha='center',
        va='center', fontsize=13, fontweight='bold', color='#1a2e4a')
ax.text(x, SEGREGACION + DOT_WITHIN + EST_WITHIN/2, f'{EST_WITHIN:.1f}%',
        ha='center', va='center', fontsize=13, fontweight='bold', color='white')

ax.axhline(y=100, color='#333', lw=0.8, ls=':', alpha=0.4, zorder=1)
ax.text(x + 0.42, 100, f'Δ = {BRECHA:.3f}', ha='left', va='center',
        fontsize=9, style='italic', color='#555')

ax.set_xlim(-0.6, 0.9)
ax.set_ylim(0, 108)
ax.set_xticks([x])
ax.set_xticklabels(['Brecha salarial\nformal – informal'], fontsize=11)
ax.set_ylabel('Proporción de la brecha salarial (%)', fontsize=10)
ax.grid(axis='y', alpha=0.25, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc='center right', fontsize=9, framealpha=0.95)
ax.set_title(
    'Figura 12. Descomposición jerárquica de la brecha salarial\n'
    'Segregación entre sub-sectores vs. diferencias dentro del sub-sector\n'
    'Sector terciario privado, CDMX y Estado de México, 2005–2025',
    fontsize=12, fontweight='bold', pad=10
)

ax.text(0.0, -0.09,
        'Nota: Descomposición between-within ponderada por factor de expansión ENOE. '
        'El componente "entre sub-sectores" mide la parte de la brecha atribuible a la\n'
        'distinta distribución de trabajadores formales e informales entre los cuatro '
        'bloques SCIAN. Los componentes "dentro" replican la lógica Oaxaca-Blinder al\n'
        'interior de cada bloque. Referencia: coeficientes del segmento formal.',
        transform=ax.transAxes, fontsize=7.5, color='#555',
        verticalalignment='top', wrap=True)

plt.savefig('FIGURA_12_descomp_jerarquica.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Figura 12 guardada.")