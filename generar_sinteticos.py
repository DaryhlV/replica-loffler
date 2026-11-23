"""Genera un corpus SINTÉTICO de cláusulas de términos y condiciones en español (permitido por el curso).

El dataset chileno de Löffler et al. (2025) no es público, así que se construye uno con la misma
estructura: cláusulas justas y cláusulas oscuras (Art. 16 LPC) con 6 clases, ~20 % abusivas.
Las cláusulas imitan la redacción de contratos reales: las abusivas son sutiles y las justas usan el
mismo vocabulario (modificar, suspender, responsabilidad), para que no haya pistas obvias.
No son cláusulas reales de ninguna empresa.

    python generar_sinteticos.py --n 1000
"""
import argparse
import random

import pandas as pd

EMP = ["la Empresa", "el Banco", "la Entidad", "el Proveedor", "la Plataforma", "la Compañía"]
SERV = ["el servicio", "la aplicación", "la cuenta", "la banca móvil", "la billetera digital", "la tarjeta"]
USU = ["el Usuario", "el Cliente", "el titular", "el consumidor"]
CANAL = ["su sitio web", "sus canales digitales", "la aplicación móvil", "su página web oficial"]
DIAS = [15, 30, 45, 60]
CORTO = [3, 5, 7]

# Plantillas de cláusulas abusivas, agrupadas por clase
ABUSIVAS = {
    "cr": [
        "Las modificaciones a las presentes condiciones entrarán en vigencia desde su publicación en {C}, y el uso posterior de {V} por parte de {U} implicará su aceptación.",
        "{E} podrá actualizar el contenido de este contrato cuando lo considere necesario para la mejor prestación de {V}; dichas actualizaciones serán aplicables desde su difusión en {C}.",
        "Se entenderá que {U} acepta las nuevas condiciones si no manifiesta su disconformidad dentro de los {DC} días siguientes a su publicación en {C}, sin que ello le otorgue el derecho a resolver el contrato.",
        "{E} se reserva la facultad de modificar las características, funcionalidades y requisitos de {V}, informando de ello mediante {C}.",
        "Las condiciones particulares de {V} podrán ser complementadas o sustituidas por {E} mediante avisos publicados en {C}, los cuales formarán parte integrante del presente contrato.",
        "Toda referencia a estas condiciones se entenderá hecha a la versión vigente publicada en {C}, la cual podrá ser revisada periódicamente por {E} sin requerir el consentimiento expreso de {U}.",
        "Las variaciones en los límites de operación de {V} que disponga {E} regirán desde el día siguiente a su comunicación por {C}.",
    ],
    "ter": [
        "{E} podrá resolver el presente contrato cuando, a su criterio, el uso de {V} no se ajuste a sus políticas internas, sin que {U} tenga derecho a compensación alguna.",
        "{E} tendrá la facultad de cerrar {V} si detecta inactividad, bastando para ello un aviso publicado en {C}.",
        "La relación contractual podrá concluir por decisión de {E}, comunicándolo a {U} con {DC} días de anticipación, sin necesidad de expresar los motivos.",
        "{E} podrá desactivar de manera definitiva el acceso de {U} a {V} por razones de orden comercial, sin que ello genere responsabilidad de su parte.",
        "En caso de que {E} decida discontinuar {V}, los saldos y beneficios acumulados por {U} quedarán sin efecto desde la fecha de cierre.",
        "{E} podrá suspender {V} mientras realiza las verificaciones que estime pertinentes, sin un plazo máximo para su conclusión.",
    ],
    "ch": [
        "Las tasas de interés, comisiones y gastos aplicables a {V} podrán ser ajustados por {E} conforme a las condiciones del mercado, siendo comunicados a través de {C}.",
        "{E} podrá incorporar nuevas comisiones por servicios asociados a {V}, las cuales se cargarán en la cuenta de {U} desde su publicación en el tarifario.",
        "El costo de membresía de {V} podrá ser reajustado anualmente según los parámetros que determine {E}.",
        "Las tarifas aplicables son las que figuran en {C} al momento de cada operación, aun cuando difieran de las informadas al momento de la contratación.",
        "{U} autoriza a {E} a cargar en su cuenta cualquier gasto adicional vinculado a la administración de {V} que se genere durante la vigencia del contrato.",
        "El tipo de cambio aplicable a las operaciones en moneda extranjera será el que {E} fije el día del cargo, sin que {U} pueda consultarlo previamente.",
    ],
    "er": [
        "Las operaciones registradas en los sistemas de {E} se presumirán correctas, correspondiendo a {U} acreditar cualquier error en su procesamiento.",
        "Los cargos duplicados o indebidos originados en incidencias técnicas serán regularizados solo si {U} los reporta dentro de los {DC} días siguientes a su realización.",
        "{E} podrá debitar de la cuenta de {U} los montos abonados por error, así como los gastos derivados de dicha regularización.",
        "Las diferencias que se produzcan por fallas en la conversión de moneda efectuada por {E} serán asumidas por {U}.",
        "Si por una incidencia en la plataforma de {E} una operación no se completa, {U} deberá iniciarla nuevamente y asumir las comisiones que correspondan.",
        "La información de saldos mostrada en {V} es referencial, por lo que {E} no se encontrará en la obligación de respetar los montos mostrados por error.",
    ],
    "ltd": [
        "{E} no asume responsabilidad por la interrupción de {V} originada en fallas de los sistemas de comunicación, incluidas aquellas vinculadas a su propia infraestructura tecnológica.",
        "La responsabilidad de {E} por cualquier perjuicio derivado de {V} se limitará al monto de las comisiones cobradas a {U} en los últimos tres meses.",
        "{E} no será responsable por las operaciones efectuadas con las credenciales de {U}, cualquiera sea la circunstancia en que hayan sido obtenidas por terceros.",
        "{U} reconoce que el uso de {V} se realiza bajo su propio riesgo, y que {E} no garantiza su disponibilidad ni la exactitud de la información mostrada.",
        "{E} no responderá por los daños que sufra {U} como consecuencia de demoras en la ejecución de sus instrucciones, cualquiera sea su causa.",
        "En ningún caso {E} será responsable por el lucro cesante o el daño indirecto que {U} pudiera sufrir por el uso de {V}.",
    ],
    "nod": [
        "Cualquier observación al estado de cuenta deberá formularse dentro de los {DC} días siguientes a su recepción; vencido dicho plazo, se entenderá que {U} lo acepta en su totalidad.",
        "Los reclamos relacionados con {V} solo podrán ser presentados mediante el formulario disponible en {C}, no siendo admitidos por otros medios.",
        "Antes de acudir a cualquier autoridad, {U} deberá agotar el procedimiento interno de atención de {E}, cuyo resultado tendrá carácter definitivo.",
        "{U} declara conocer y aceptar que las grabaciones y registros de {E} constituirán la única prueba válida de las operaciones realizadas.",
        "Las controversias derivadas del presente contrato se someterán a arbitraje en la ciudad que designe {E}, renunciando {U} al fuero de su domicilio.",
        "{U} renuncia a solicitar la devolución de los montos debitados una vez transcurridos {DC} días desde la fecha del cargo.",
    ],
}
# Plantillas de cláusulas justas (no abusivas), con vocabulario parecido al de las abusivas
JUSTAS = [
    "Las modificaciones a las presentes condiciones serán comunicadas a {U} mediante {C} con {D} días de anticipación a su entrada en vigencia; si no está de acuerdo, {U} podrá resolver el contrato sin penalidad.",
    "{E} podrá actualizar el contenido de este contrato para adecuarlo a cambios normativos, informando a {U} con {D} días de anticipación y respetando su derecho a darlo por terminado.",
    "Las mejoras en las funcionalidades de {V} que no impliquen nuevos costos ni obligaciones para {U} podrán implementarse con su sola publicación en {C}.",
    "La versión vigente de estas condiciones se encuentra disponible en {C}, y {U} podrá solicitar una copia en cualquier momento.",
    "{E} podrá resolver el presente contrato si {U} utiliza {V} para fines ilícitos, comunicándole previamente los motivos y otorgándole un plazo para presentar sus descargos.",
    "Si {V} permanece sin movimientos durante doce meses, {E} lo comunicará a {U} con {D} días de anticipación antes de proceder a su cierre, devolviendo el saldo disponible.",
    "{U} podrá dar por terminado el contrato en cualquier momento a través de {C}, sin expresión de causa y sin costo alguno.",
    "En caso de que {E} decida discontinuar {V}, lo comunicará con {D} días de anticipación y transferirá los saldos a la cuenta que {U} indique.",
    "{E} podrá bloquear de forma preventiva y sin previo aviso las operaciones que presenten indicios de fraude, informando a {U} dentro de las 24 horas siguientes y restableciendo el servicio una vez verificada la operación.",
    "{E} podrá suspender temporalmente {V} por mantenimiento programado, informándolo a {U} con anticipación a través de {C}.",
    "Las tasas de interés, comisiones y gastos aplicables a {V} figuran en la hoja resumen entregada a {U}, y cualquier incremento será comunicado con {D} días de anticipación.",
    "Las comisiones vigentes se encuentran publicadas en el tarifario de {E} y en {C}, y no se cobrarán conceptos que no hayan sido informados previamente a {U}.",
    "Si {E} incrementa el costo de membresía de {V}, {U} podrá resolver el contrato antes de su aplicación sin pagar penalidad alguna.",
    "El tipo de cambio aplicable a las operaciones en moneda extranjera se mostrará a {U} antes de que confirme cada operación en {V}.",
    "{E} reembolsará a {U} los cargos duplicados o indebidos que se originen en errores de sus sistemas, dentro de un plazo máximo de {D} días.",
    "Cuando {E} abone por error un monto en la cuenta de {U}, se lo comunicará antes de efectuar la regularización correspondiente.",
    "Las incidencias técnicas que impidan completar una operación en {V} no generarán comisiones para {U}.",
    "{E} responderá por los daños causados a {U} cuando se deban a fallas de su infraestructura tecnológica o a su culpa.",
    "{E} no será responsable por los perjuicios derivados del incumplimiento, por parte de {U}, de las medidas de seguridad informadas, salvo que medie culpa de {E}.",
    "{E} asumirá las operaciones no reconocidas realizadas después de que {U} haya reportado la pérdida o sustracción de sus credenciales.",
    "{E} no responderá por la interrupción de {V} debida a casos fortuitos o de fuerza mayor debidamente acreditados, y procurará restablecer el servicio a la brevedad.",
    "Los reclamos podrán presentarse en agencias, por teléfono o mediante {C}, y {E} los responderá en un plazo máximo de {D} días, sin perjuicio del derecho de {U} a acudir a la autoridad competente.",
    "{U} podrá observar los cargos de su estado de cuenta dentro de los plazos establecidos por la normativa vigente.",
    "Las controversias derivadas del presente contrato podrán ser presentadas por {U} ante los tribunales de su domicilio o ante la autoridad de protección al consumidor.",
    "Las grabaciones de las llamadas podrán ser utilizadas como medio de prueba, y {U} podrá solicitar una copia de ellas.",
    "Para efectos del presente contrato, se entiende por {V} el canal que permite a {U} realizar consultas y operaciones.",
    "{U} es responsable de mantener la confidencialidad de su clave y de no compartirla con terceros.",
    "Para acceder a {V}, {U} deberá ser mayor de edad y contar con un documento de identidad vigente.",
    "Las transferencias realizadas fuera del horario bancario se procesarán el siguiente día hábil.",
    "{U} declara que la información proporcionada al registrarse en {V} es verdadera y se compromete a mantenerla actualizada.",
    "Los datos personales de {U} serán tratados conforme a la política de privacidad publicada en {C}.",
    "{U} podrá descargar sus estados de cuenta de los últimos doce meses desde {V}.",
    "{E} enviará a {U} una notificación por cada operación realizada, al correo o teléfono que haya registrado.",
    "Estos términos se rigen por las leyes de la República del Perú.",
    "{U} podrá elegir si desea recibir publicidad de {E} y retirar su consentimiento cuando lo desee.",
    "Las operaciones que superen el límite diario requerirán un factor adicional de autenticación.",
    "{U} podrá configurar los límites de sus operaciones desde la sección de seguridad de {V}.",
    "{E} podrá solicitar a {U} información adicional para validar operaciones inusuales, conforme a la normativa de prevención de lavado de activos.",
]


PREF = ["", "", "", "", "Asimismo, ", "Sin perjuicio de lo anterior, ", "Para tal efecto, ", "Adicionalmente, ", "De otro lado, "]


def rellenar(t, rnd):
    txt = t.format(E=rnd.choice(EMP), V=rnd.choice(SERV), U=rnd.choice(USU), C=rnd.choice(CANAL),
                   D=rnd.choice(DIAS), DC=rnd.choice(CORTO))
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
    n_test = int(0.3 * a.n)
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
    df.loc[df["parte"] == "dev", "split"] = "train"
    # Toma el 30 % de desarrollo como validación, por separado en abusivas y justas
    for abusiva in (True, False):
        grupo = df[(df["parte"] == "dev") & (df["etiquetas"].str.len().gt(0) == abusiva)].index
        df.loc[grupo[: round(0.3 * len(grupo))], "split"] = "val"
    df.drop(columns="parte").to_csv("data/sintetico.csv", index=False, encoding="utf-8")
    ab = df["etiquetas"].str.len().gt(0)
    print(f"{len(df)} cláusulas sintéticas ({100 * ab.mean():.1f} % abusivas) → data/sintetico.csv")
    print("La mitad de las cláusulas de prueba usa redacciones (plantillas) que no aparecen en entrenamiento.")
    print(df.groupby("split")["etiquetas"].apply(lambda e: f"{len(e)} ({100 * e.str.len().gt(0).mean():.0f} % abusivas)").to_string())


if __name__ == "__main__":
    main()
