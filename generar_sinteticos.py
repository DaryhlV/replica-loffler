"""Genera un corpus SINTÉTICO de cláusulas de términos y condiciones en español (permitido por el curso).

El dataset chileno de Löffler et al. (2025) no es público, así que se construye uno con la misma
estructura: cláusulas justas y cláusulas oscuras (Art. 16 LPC) con 6 clases, ~20 % abusivas.
No son cláusulas reales de ninguna empresa.

    python generar_sinteticos.py --n 1000
"""
import argparse
import random

import pandas as pd

EMP = ["la Empresa", "el Banco", "la Entidad", "el Proveedor", "la Plataforma", "la Compañía"]
SERV = ["el servicio", "la aplicación", "la cuenta", "la banca móvil", "la billetera digital", "la plataforma"]
USU = ["el Usuario", "el Cliente", "el titular", "el consumidor"]
MOD = ["en cualquier momento", "a su sola discreción", "sin expresión de causa", "cuando lo estime conveniente", "sin necesidad de justificación"]
SINAV = ["sin previo aviso", "sin necesidad de notificación", "sin comunicarlo previamente", "sin aviso alguno"]

# Plantillas de cláusulas abusivas, agrupadas por clase
ABUSIVAS = {
    "cr": [
        "{E} podrá modificar los presentes términos y condiciones {M} {S}, y {U} no podrá oponerse a dichos cambios.",
        "{E} se reserva el derecho de cambiar estas condiciones {M}; los cambios serán obligatorios para {U} desde su publicación.",
        "Las condiciones de {V} podrán ser alteradas unilateralmente por {E} {S}, entendiéndose aceptadas por {U}.",
        "{E} podrá reemplazar total o parcialmente este contrato {M}, sin que {U} tenga derecho a reclamo.",
        "Cualquier cambio en estos términos que decida {E} regirá desde el momento en que se publique en {V}, sin que se requiera la aceptación de {U}.",
        "Al seguir usando {V}, {U} acepta automáticamente cualquier modificación que {E} introduzca {M}.",
    ],
    "ter": [
        "{E} podrá suspender o cancelar {V} {M} {S}.",
        "{E} se reserva el derecho de bloquear o cerrar {V} de {U} {M}, sin responsabilidad alguna.",
        "{E} podrá dar por terminado el contrato {M} y {S}, sin que ello genere indemnización a favor de {U}.",
        "El acceso a {V} podrá ser interrumpido definitivamente por {E} {M}.",
        "{E} tiene la facultad de dejar sin efecto la relación con {U} y desactivar {V} {S}, por cualquier motivo.",
    ],
    "ch": [
        "{E} podrá modificar las tarifas, comisiones y cargos de {V} {M} {S}.",
        "Los precios y comisiones podrán ser incrementados por {E} {M}, aplicándose de inmediato a {U}.",
        "{E} se reserva la facultad de fijar nuevas comisiones por el uso de {V} {M}.",
        "El costo de mantenimiento de {V} podrá ser reajustado por {E} {S}, y se cargará automáticamente a {U}.",
    ],
    "er": [
        "{U} asumirá las consecuencias de cualquier error en el procesamiento de operaciones atribuible a los sistemas de {E}.",
        "{E} no estará obligada a revertir cargos originados en fallas de sus propios sistemas, los que serán de cargo de {U}.",
        "Cualquier error administrativo o informático de {E} en {V} será soportado exclusivamente por {U}.",
        "Si por un fallo técnico de {E} una transferencia se duplica, {U} deberá cubrir el monto y las comisiones generadas.",
    ],
    "ltd": [
        "{E} no será responsable por ningún daño, pérdida o perjuicio derivado del uso de {V}, cualquiera sea su causa.",
        "En la máxima medida permitida por la ley, {E} queda liberada de toda responsabilidad por fallas, interrupciones o fraudes en {V}.",
        "{E} no responderá por operaciones no reconocidas realizadas a través de {V}, aun cuando se deban a fallas de seguridad propias.",
        "La responsabilidad de {E} frente a {U} se limita en todo caso al monto de las comisiones pagadas en el último mes.",
    ],
    "nod": [
        "{U} renuncia a presentar reclamos ante cualquier autoridad administrativa o judicial por el uso de {V}.",
        "Todo reclamo deberá presentarse por escrito en el domicilio de {E} dentro de las 24 horas siguientes; en caso contrario, se entenderá que {U} renuncia a su derecho.",
        "{U} solo podrá reclamar a través de los canales que {E} determine, y no podrá recurrir a otras instancias.",
        "{U} acepta que la única vía para resolver controversias es el procedimiento interno de {E}, cuya decisión será definitiva.",
    ],
}
# Plantillas de cláusulas justas (no abusivas)
JUSTAS = [
    "{U} podrá solicitar en cualquier momento la copia de su contrato a través de {V}.",
    "{E} informará a {U} con {D} días de anticipación cualquier cambio en las comisiones, y {U} podrá resolver el contrato sin penalidad.",
    "Los reclamos podrán presentarse en las agencias, por teléfono o en {V}, y {E} los responderá en un plazo de {D} días.",
    "{U} puede activar o desactivar las notificaciones de {V} desde la sección de configuración.",
    "Si {E} decide cerrar {V}, lo comunicará a {U} con {D} días de anticipación y devolverá el saldo disponible.",
    "{E} responde por las operaciones no reconocidas cuando {U} haya reportado oportunamente el bloqueo de su tarjeta.",
    "El horario de atención de los canales presenciales de {E} es de lunes a viernes.",
    "{U} podrá elegir si desea recibir publicidad de {E} y retirar su consentimiento cuando quiera.",
    "Los datos de {U} se conservarán solo durante el tiempo necesario para cumplir la finalidad informada.",
    "{E} enviará el estado de cuenta de {V} de forma mensual y gratuita.",
    "{U} podrá consultar sus movimientos y saldos las 24 horas del día a través de {V}.",
    "{E} comunicará cualquier modificación de estas condiciones con al menos 30 días de anticipación, y {U} podrá terminar el contrato sin costo si no está de acuerdo.",
    "{U} es responsable de mantener la confidencialidad de su clave y de no compartirla con terceros.",
    "Para acceder a {V}, {U} deberá ser mayor de 18 años y contar con un documento de identidad vigente.",
    "{U} podrá cerrar {V} en cualquier momento a través de los canales de atención, sin costo alguno.",
    "{E} atenderá los reclamos de {U} en un plazo máximo de 15 días hábiles, sin perjuicio de su derecho a acudir a la autoridad competente.",
    "Las comisiones vigentes se encuentran publicadas en el tarifario de {E} y en {V}.",
    "{U} puede ejercer sus derechos de acceso, rectificación, cancelación y oposición escribiendo a los canales de {E}.",
    "{E} adopta medidas de seguridad razonables para proteger la información de {U}.",
    "En caso de pérdida o robo del celular, {U} deberá bloquear {V} comunicándose con {E}.",
    "{E} podrá suspender temporalmente {V} por mantenimiento programado, informándolo con anticipación a {U}.",
    "Estos términos se rigen por las leyes de la República del Perú.",
    "{E} enviará a {U} una notificación por cada operación realizada con su tarjeta.",
    "Las transferencias realizadas fuera del horario bancario se procesarán el siguiente día hábil.",
    "{U} declara que la información proporcionada al registrarse en {V} es verdadera.",
    "{E} reembolsará a {U} los cargos que se demuestren originados en errores de sus sistemas.",
    "La presente política describe qué datos personales recopila {E} y con qué finalidad.",
    "{U} podrá descargar sus estados de cuenta de los últimos doce meses desde {V}.",
    "{E} responderá por los daños causados a {U} cuando se deban a su culpa o negligencia.",
    "Cualquier modificación de las tarifas será informada con 45 días de anticipación, y {U} podrá dar por terminado el contrato.",
]


PREF = ["", "", "", "Asimismo, ", "Por otro lado, ", "En ese sentido, ", "Adicionalmente, ", "De igual forma, "]


def rellenar(t, rnd):
    txt = t.format(E=rnd.choice(EMP), V=rnd.choice(SERV), U=rnd.choice(USU), M=rnd.choice(MOD),
                   S=rnd.choice(SINAV), D=rnd.choice([15, 30, 45, 60]))
    for a_, b_ in ((" a el ", " al "), (" de el ", " del ")):
        txt = txt.replace(a_, b_)
    p = rnd.choice(PREF)
    txt = p + txt[0].lower() + txt[1:] if p else txt
    return txt[0].upper() + txt[1:]


# Separa las plantillas en las que pueden usarse en entrenamiento y las reservadas para prueba
def reservar(plantillas, rnd, frac=0.25):
    """Separa plantillas solo-entrenamiento y solo-prueba, para medir generalización a redacciones nuevas."""
    idx = list(range(len(plantillas)))
    rnd.shuffle(idx)
    k = max(1, round(frac * len(idx)))
    return idx[k:], idx[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--abusivas", type=float, default=0.2, help="proporción abusiva (paper: 20.3 %%)")
    ap.add_argument("--semilla", type=int, default=42)
    a = ap.parse_args()
    rnd = random.Random(a.semilla)
    plan = {c: reservar(t, rnd) for c, t in ABUSIVAS.items()}
    plan["justa"] = reservar(JUSTAS, rnd)
    n_test = int(0.2 * a.n)
    filas, vistos = [], set()

    def una(abusiva, parte):
        # Crea una cláusula; en prueba, la mitad usa plantillas reservadas
        uso = 1 if parte == "test" and rnd.random() < 0.5 else 0
        if abusiva:
            clases = [rnd.choice(list(ABUSIVAS))]
            if rnd.random() < 0.15:
                clases.append(rnd.choice([c for c in ABUSIVAS if c not in clases]))
            texto = " ".join(rellenar(ABUSIVAS[c][rnd.choice(plan[c][uso])], rnd) for c in clases)
            return texto, clases
        return rellenar(JUSTAS[rnd.choice(plan["justa"][uso])], rnd), []

    # Genera las cláusulas de prueba y de desarrollo con su proporción de abusivas
    for parte, total in (("test", n_test), ("dev", a.n - n_test)):
        n_ab, hechos, intentos = round(a.abusivas * total), 0, 0
        while hechos < total and intentos < 100000:
            intentos += 1
            abusiva = hechos < n_ab
            texto, clases = una(abusiva, parte)
            if texto in vistos:
                continue
            vistos.add(texto)
            filas.append({"texto": texto, "etiquetas": ";".join(clases), "parte": parte})
            hechos += 1
    df = pd.DataFrame(filas).sample(frac=1, random_state=a.semilla).reset_index(drop=True)
    df["split"] = "test"
    dev = df[df["parte"] == "dev"].index
    df.loc[dev, "split"] = "train"
    df.loc[dev[: len(dev) // 8], "split"] = "val"
    df.drop(columns="parte").to_csv("data/sintetico.csv", index=False, encoding="utf-8")
    ab = df["etiquetas"].str.len().gt(0)
    print(f"{len(df)} cláusulas sintéticas ({100 * ab.mean():.1f} % abusivas) → data/sintetico.csv")
    print("La mitad de las cláusulas de prueba usa redacciones (plantillas) que no aparecen en entrenamiento.")
    print(df.groupby("split")["etiquetas"].apply(lambda e: f"{len(e)} ({100 * e.str.len().gt(0).mean():.0f} % abusivas)").to_string())


if __name__ == "__main__":
    main()
