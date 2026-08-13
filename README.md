# Brecha salarial formal–informal en el Sector terciario privado, CDMX y Estado de México, 2005–2025

Código de replicación de la tesis de licenciatura en Ingeniería Económica y Financiera.

**Autor:** Luis Samuel Barria Gutiérrez
**Director:** Dr. Luis Alberto Quezada
**Institución:** Universidad Autónoma del Estado de Hidalgo — Escuela Superior de Apan

---

## Pregunta de investigación

¿En qué medida las diferencias salariales entre trabajadores formales e informales reflejan diferencias en el capital humano, y en qué medida reflejan diferencias en la forma en que el mercado laboral remunera ese mismo capital humano?

## Datos

| | |
|---|---|
| Fuente | ENOE, microdatos trimestrales (INEGI) — tabla SDEM |
| Cobertura | Ciudad de México y Estado de México |
| Periodo | 2005-T1 a 2025-T4, 83 trimestres |
| Exclusión | 2020-T2 (levantamiento telefónico no comparable) |
| Universo | Trabajadores subordinados y remunerados del sector terciario privado |
| Observaciones | 245,508 |
| Deflactor | INPC trimestral, base 2018 = 100 |

Los microdatos de la ENOE son de acceso público y gratuito en el sitio de INEGI. 
## Definiciones operativas

- **Formalidad**: `emp_ppal == 2` (condición de informalidad de la ENOE, matriz Hussmanns-INEGI). El tipo de contrato y las prestaciones **no** entran en la definición; se usan únicamente como validación descriptiva del grupo clasificado.
- **Escolaridad**: `anios_esc` de la ENOE. Se excluye el código 99 (*no especificado*), que corresponde a 340 observaciones.
- **Experiencia potencial**: `edad − años de escolaridad − 6`, truncada en cero.
- **Salario**: ingreso por hora real, winsorizado P1–P99.
- **Ponderación**: factor de expansión de la ENOE en todas las estimaciones y en todas las medias.

## Especificación

Ecuación de Mincer estimada por segmento:

```
ln(w) = β₀ + β₁·escolaridad + β₂·experiencia + β₃·experiencia² + Σ δₜ·D_año + ε
```

- WLS ponderado por factor de expansión
- Errores robustos HC3
- Efectos fijos de año, `Treatment(reference=2005)`
- Descomposición de Oaxaca-Blinder con coeficientes formales como referencia; componente estructural obtenido como residual

## Resultados principales

| Resultado | Valor |
|---|---|
| Brecha logarítmica | 0.4297 |
| Descomposición estándar | 82.7 % dotaciones / 17.3 % estructural |
| Descomposición jerárquica | 47.7 % entre bloques / 39.1 % dotaciones intra / 13.3 % estructural intra |
| Retorno a la escolaridad | 11.83 % formal vs. 5.03 % informal (razón 2.35) |
| Gradiente educativo | −5.1 % (sin primaria) a +52.1 % (superior) |
| Tasa de formalidad | 46.2 % |

Las dos descomposiciones **no están anidadas y no compiten**: la estándar parte la brecha tratando la ubicación sectorial como una característica más del vector de dotaciones; la jerárquica la estima a nivel de bloque y le asigna un término propio. Responden preguntas distintas.

---

## Orden de ejecución

Los scripts de `pipeline/` están numerados en orden de ejecución. Los nombres originales se conservan después del prefijo para que correspondan con los que se citan en el documento.

| # | Script | Qué produce |
|---|---|---|
| 00 | `exploracion_variables_enoe.py` | Diagnóstico de las variables educativas de la ENOE en cuatro trimestres del periodo. No modifica datos. |
| 01 | `construccion_base.py` | **Pipeline maestro.** Apila los 83 trimestres, construye las variables de capital humano, deflacta con INPC y genera `BASE_FINAL_CORREGIDA_2025.csv`. |
| 02 | `winsorizacion.py` | Winsorización P1–P99 del ingreso real por hora. |
| 03 | `auditoria_integridad.py` | Auditoría de integridad. Verifica que excluir `esc_anios == 99` devuelve exactamente 245,508 observaciones. |
| 04 | `oaxaca_global_ponderado.py` | Descomposición O-B global con ponderación consistente en coeficientes y medias. |
| 05 | `oaxaca_global_oficial.py` | Descomposición O-B global bajo la especificación oficial, comparable con la sectorial. Produce 82.7 / 17.3. |
| 06 | `mincer_oaxaca_por_bloque.py` | Mincer y O-B dentro de cada bloque sectorial SCIAN. |
| 07 | `descomposicion_jerarquica.py` | Partición jerárquica *between–within*. Produce 47.7 / 39.1 / 13.3. |
| 08 | `brecha_por_tramo_educativo.py` | Brecha por tramo educativo, global y por bloque. |
| 09 | `verificacion_integral.py` | **Verificación integral.** Regenera todas las cifras del Capítulo 5 (Anexo B). |
| 10 | `robustez_anexo_c.py` | Robustez C.1–C.3: winsorización, convención de referencia, población objetivo. |
| 11 | `robustez_muestra_asalariados.py` | Comparación muestra completa vs. solo asalariados. |
| 12 | `figuras_capitulo5.py` | Figuras 1 a 11. |
| 13 | `figura_descomposicion_jerarquica.py` | Figura 12. |

Cada archivo lleva el prefijo numérico de su posición (por ejemplo, `07_descomposicion_jerarquica.py`). Todos abren con un encabezado que declara qué hace, qué especificación emplea, qué recibe y qué produce.

Para reproducir solo las cifras del documento sin reconstruir la base, basta ejecutar **09**.

## Requisitos

```
python >= 3.9
pandas
numpy
statsmodels
scipy
matplotlib
```

```bash
pip install -r requirements.txt
```

## Rutas

Los scripts se escribieron con rutas absolutas de Windows, en dos equipos distintos. Antes de ejecutar, edite en cada script la constante de ruta al inicio (`RUTA`, `ARCHIVO`, `RUTA_BASE` o `RUTA_CSV`) y apúntela a la carpeta local con los datos. `config.py` contiene la ruta única sugerida.


## Advertencia sobre interpretación

Las estimaciones son descriptivas. El componente estructural de la descomposición **no** identifica un efecto causal de la informalidad sobre el salario: recoge diferencias en la estructura de remuneración entre segmentos que pueden originarse en variables no observadas, en selección no aleatoria hacia cada segmento, o en segmentación institucional. El documento discute estas limitaciones en su sección de limitaciones metodológicas.


