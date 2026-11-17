"""Cláusulas oscuras (Dark Clauses) del esquema de Löffler et al. (2025), Tabla 1.

Son las cláusulas que la Ley de Protección al Consumidor chilena (Art. 16 LPC) presume abusivas.
Equivalen a las "cláusulas de riesgo" que busca detectar la tesis.
"""

# [PASO 3] Esquema de anotación: 6 cláusulas oscuras (Art. 16 LPC), Löffler et al. (2025), Tabla 1
CLASES = {
    "cr":  ("Modificación unilateral", "modificación unilateral y arbitraria de los términos sin que el consumidor pueda rechazarla"),
    "ter": ("Terminación unilateral", "terminación unilateral y arbitraria del servicio"),
    "ch":  ("Cambio de precio", "modificación unilateral del precio del servicio"),
    "er":  ("Errores del proveedor", "el consumidor asume las consecuencias de errores administrativos o internos del proveedor"),
    "ltd": ("Limitación de responsabilidad", "limitación de la responsabilidad del proveedor"),
    "nod": ("Limita derechos", "limita sustancialmente la forma en que el consumidor puede ejercer sus derechos"),
}
CLAVES = list(CLASES)
DETECCION = ["no abusiva", "abusiva"]
