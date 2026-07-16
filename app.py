import asyncio
import base64
import os
import re
import sqlite3
import time
import unicodedata
import uuid
import httpx
import azure.cognitiveservices.speech as speechsdk
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Carga las variables definidas en el archivo .env (junto a este app.py)
# hacia el entorno, para no tener que escribir la API key en el código
# ni tener que exportarla a mano cada vez que abres una terminal nueva.
load_dotenv()

# SDK oficial de Groq (compatible con el formato de OpenAI por debajo).
# pip install groq
from groq import Groq

# --- Rutas base, RELATIVAS a este archivo -----------------------------------
# Esto es a propósito: así funciona igual en la laptop con Windows del profe,
# en Mac o en Linux, sin tener que cambiar nada a mano.
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
RUTA_DB = os.path.join(DIRECTORIO_ACTUAL, "database.db")
RUTA_VIDEOS = os.path.join(DIRECTORIO_ACTUAL, "videos")
RUTA_FRONTEND = os.path.join(DIRECTORIO_ACTUAL, "frontend")
RUTA_MODELOS = os.path.join(DIRECTORIO_ACTUAL, "models")

# Objeto principal de la aplicación: aquí se registran todas las rutas (endpoints).
app = FastAPI(title="La Teacher MarIA - Bot")


def inicializar_base_datos():
    """
    Crea la carpeta de videos y la base de datos si no existen.
    Si la tabla está vacía, mete un registro de prueba para que la demo
    no se vea en blanco desde el primer arranque.
    """
    os.makedirs(RUTA_VIDEOS, exist_ok=True)

    conexion = sqlite3.connect(RUTA_DB)
    cursor = conexion.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS videos (
            id_video INTEGER PRIMARY KEY AUTOINCREMENT,
            ruta_archivo TEXT NOT NULL,
            grado TEXT,
            materia TEXT,
            subtema TEXT,
            duracion INTEGER,
            descripcion TEXT,
            fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute("SELECT COUNT(*) FROM videos")
    if cursor.fetchone()[0] == 0:
        cursor.execute('''
            INSERT INTO videos (ruta_archivo, grado, materia, subtema, duracion, descripcion)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            "video_001.mp4",
            "3ro Primaria",
            "Matemáticas",
            "Fracciones",
            120,
            "Introducción visual a las fracciones simples para niños."
        ))
        print("Aviso: la base de datos estaba vacía, se insertó un registro de prueba.")
        print(f"Coloca un archivo llamado 'video_001.mp4' dentro de: {RUTA_VIDEOS}")

    conexion.commit()
    conexion.close()


# Se ejecuta una sola vez, al arrancar el servidor (no en cada request).
inicializar_base_datos()

# Montamos la carpeta de videos como archivos estáticos.
# Como ya corrimos inicializar_base_datos() antes, la carpeta seguro existe
# (si no, StaticFiles truena al arrancar el servidor).
app.mount("/videos", StaticFiles(directory=RUTA_VIDEOS), name="videos")

# Modelos de face-api.js servidos localmente (pesan menos de 1 MB en total),
# así la demo no depende del internet del salón/oficina el día de la presentación.
app.mount("/models", StaticFiles(directory=RUTA_MODELOS), name="models")

# Imagen base del avatar de MarIA + los recortes de boca (mismo personaje,
# sacados de un video generado con Google Flow), servidos como estáticos.
app.mount("/avatar", StaticFiles(directory=os.path.join(RUTA_FRONTEND, "avatar")), name="avatar")


# --- Conexión a la base de datos ---------------------------------------------
# Abre una conexión nueva de SQLite cada vez que se llama (SQLite no soporta
# bien compartir una sola conexión entre varias peticiones al mismo tiempo).
# row_factory = sqlite3.Row permite acceder a las columnas por nombre
# (fila["subtema"]) en vez de solo por posición (fila[2]).
def obtener_conexion():
    conexion = sqlite3.connect(RUTA_DB)
    conexion.row_factory = sqlite3.Row
    return conexion


# ==============================================================================
# --- ASISTENTE DE VOZ (MarIA) -------------------------------------------------
# ==============================================================================

# Modelo Llama 3.3 70B corriendo en el hardware LPU de Groq: rápido, gratis
# (con límites diarios/por minuto) y sin los picos de saturación que traíamos
# con Gemini. Puede cambiarse por otro modelo válido de Groq si hiciera falta.
MODELO_GROQ = "llama-3.3-70b-versatile"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
cliente_groq = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

if cliente_groq is None:
    print("=" * 70)
    print("AVISO: no se encontró GROQ_API_KEY.")
    print("El asistente de voz (/api/chat) NO va a funcionar hasta que")
    print("pongas tu clave en el archivo .env dentro de la carpeta bot/.")
    print("=" * 70)

# --- Voz de MarIA: Azure AI Speech (voces neuronales) ------------------------
# La API key vive SOLO en el servidor (variable de entorno). El navegador
# nunca la ve: le pide el audio ya generado a nuestro propio endpoint
# /api/tts, y ese endpoint es el único que habla con Azure.
AZURE_SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "")
VOZ_MARIA = "es-MX-CandelaNeural"

if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
    print("=" * 70)
    print("AVISO: no se encontró AZURE_SPEECH_KEY / AZURE_SPEECH_REGION.")
    print("El endpoint /api/tts (voz de MarIA) NO va a funcionar hasta que")
    print("pongas esas variables en el archivo .env.")
    print("=" * 70)

# Memoria de conversación POR SESIÓN (una por salón/pestaña), no global.
# Con una sola sesión compartida entre todos, las conversaciones de
# distintos salones/usuarios se mezclarían entre sí. Aquí cada sesión vive
# aislada en este diccionario, identificada por un sesion_id que genera
# el navegador y manda en cada request.
#
# A diferencia del SDK de Gemini (que traía un objeto "chat" con memoria
# incluida), la API de Groq es más simple/estándar: cada llamada es
# independiente y nosotros somos quienes mandamos el historial completo de
# la conversación en cada request. Por eso aquí guardamos, por sesión, la
# LISTA de mensajes (system + cada turno de alumno/MarIA), no un objeto chat.
#
# NOTA: esto vive en memoria RAM del proceso. Para un MVP/demo está perfecto;
# si el servidor se reinicia, la memoria de las conversaciones se pierde.
sesiones_chat = {}


# Convierte la tabla 'videos' en texto plano + un set de nombres válidos,
# para dárselo a Gemini como contexto y para validar después que no invente
# nombres de archivo que no existen.
def obtener_catalogo_videos():
    """Regresa la lista de (subtema, descripcion, nombre_archivo) para dárselo a la IA."""
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    cursor.execute("SELECT subtema, descripcion, ruta_archivo FROM videos")
    filas = cursor.fetchall()
    conexion.close()

    catalogo_texto = ""
    nombres_validos = set()
    for fila in filas:
        nombre_archivo = os.path.basename(fila["ruta_archivo"])
        nombres_validos.add(nombre_archivo)
        catalogo_texto += f"- Tema: {fila['subtema']}, Archivo: {nombre_archivo}\n"

    return catalogo_texto, nombres_validos


# Arma el "system prompt": la personalidad de MarIA + el catálogo de videos
# vigente en ese momento. Se llama cada vez que se crea una sesión nueva,
# así que si agregas un video nuevo, las conversaciones NUEVAS ya lo conocen
# (las que ya estaban abiertas no se actualizan hasta que se reinicien).
def construir_instrucciones():
    catalogo_texto, _ = obtener_catalogo_videos()
    return f"""Eres La Teacher MarIA, una maestra virtual experta, paciente y amable.
Estás dando clase a un grupo de alumnos. Tus respuestas deben ser narradas en tono de
profesora, claras y concisas (máximo 2 párrafos cortos), porque tu respuesta se va a
LEER EN VOZ ALTA, así que evita listas, símbolos raros o texto que no suene natural hablado.

Catálogo de videos disponibles en tu memoria:
{catalogo_texto}

REGLA DE ORO:
Si un alumno te pregunta algo relacionado con un tema del catálogo, explícalo brevemente
y luego decide reproducir el video correspondiente. Para ordenarle al sistema que ponga
el video, DEBES incluir al final de tu respuesta EXACTAMENTE esta etiqueta:
[VIDEO: nombre_del_archivo.mp4]
Ejemplo: "El sistema solar es fascinante. Veamos este video para entenderlo mejor. [VIDEO: video_001.mp4]"

Si preguntan algo que no está en el catálogo, explícalo con tus propias palabras y anímalos
a seguir aprendiendo, sin usar la etiqueta. Nunca digas que eres una Inteligencia Artificial."""


# Forma esperada del JSON que manda el navegador al preguntarle algo a MarIA.
class MensajeAlumno(BaseModel):
    texto: str
    sesion_id: str | None = None  # si no manda una, se crea una nueva


@app.post("/api/chat")
async def conversar(mensaje: MensajeAlumno):
    """
    Recibe la pregunta (transcrita del micrófono en el navegador), la agrega
    al historial de la sesión correspondiente (con memoria de lo ya
    platicado en ESA sesión) y decide si toca reproducir un video.
    """
    if cliente_groq is None:
        raise HTTPException(
            status_code=500,
            detail="Falta configurar la variable de entorno GROQ_API_KEY."
        )

    sesion_id = mensaje.sesion_id or str(uuid.uuid4())

    # Si es una sesión nueva, arrancamos el historial con el system prompt.
    if sesion_id not in sesiones_chat:
        sesiones_chat[sesion_id] = [
            {"role": "system", "content": construir_instrucciones()}
        ]

    historial = sesiones_chat[sesion_id]
    historial.append({"role": "user", "content": mensaje.texto})

    # Reintentamos hasta 3 veces ante errores temporales (servidores de Groq
    # saturados, o límite de peticiones por minuto/día alcanzado). El SDK de
    # Groq ya reintenta automáticamente 2 veces por su cuenta en varios de
    # estos casos, así que esto es una capa extra de seguridad encima.
    intentos_maximos = 3
    ultimo_error = None

    for intento in range(1, intentos_maximos + 1):
        try:
            respuesta = cliente_groq.chat.completions.create(
                model=MODELO_GROQ,
                messages=historial,
            )
            texto_maria = respuesta.choices[0].message.content or ""

            video_detectado = None
            if "[VIDEO:" in texto_maria:
                _, nombres_validos = obtener_catalogo_videos()
                partes = texto_maria.split("[VIDEO:")
                texto_maria = partes[0].strip()
                nombre_propuesto = partes[1].replace("]", "").strip()

                # Validamos contra el catálogo real antes de mandarlo al frontend.
                # Si la IA "alucina" un nombre de archivo que no existe, mejor
                # no reproducir nada a reproducir un video equivocado o roto.
                if nombre_propuesto in nombres_validos:
                    video_detectado = nombre_propuesto

            # Guardamos la respuesta en el historial de la sesión, para que
            # la siguiente pregunta del mismo alumno tenga memoria de esto.
            # Guardamos el texto CON la etiqueta [VIDEO:...] tal como salió,
            # así el modelo mantiene coherencia de qué video ya mostró antes.
            historial.append({"role": "assistant", "content": respuesta.choices[0].message.content or ""})

            return JSONResponse(content={
                "respuesta": texto_maria,
                "video": video_detectado,
                "sesion_id": sesion_id,
                "error": False,
            })
        except Exception as e:
            ultimo_error = e
            # "rate_limit"/"429" = límite de peticiones por minuto/día alcanzado.
            # "503"/"internal" = servidores de Groq con problemas momentáneos.
            es_error_temporal = any(
                clave in str(e).lower() for clave in ["rate_limit", "429", "503", "internal", "timeout"]
            )
            print(f"Error en Groq (intento {intento}/{intentos_maximos}): {e}")

            if es_error_temporal and intento < intentos_maximos:
                time.sleep(1.5 * intento)  # espera un poco más largo en cada reintento
                continue
            break

    # Si de plano no se pudo, quitamos la pregunta del historial (no se llegó
    # a responder), para que no quede un turno de alumno sin su respuesta.
    historial.pop()

    print(f"Se agotaron los reintentos. Último error: {ultimo_error}")
    return JSONResponse(content={
        "respuesta": "Chicos, denme un segundo, estoy organizando mis ideas.",
        "video": None,
        "sesion_id": sesion_id,
        "error": True,  # el frontend usa esto para ofrecer el botón de "reintentar"
    })


def texto_a_ssml(texto: str) -> str:
    """
    Arma el SSML que espera Azure Speech, escapando los caracteres especiales
    de XML del texto (que puede traer &, <, >, comillas, etc. si MarIA los usó).
    """
    texto_escapado = (
        texto.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
    return (
        "<speak version='1.0' xml:lang='es-MX'>"
        f"<voice xml:lang='es-MX' xml:gender='Female' name='{VOZ_MARIA}'>"
        "<prosody rate='+8%'>"
        f"{texto_escapado}"
        "</prosody>"
        "</voice></speak>"
    )


@app.get("/api/tts")
async def sintetizar_voz(texto: str):
    """
    Manda el texto de la respuesta de MarIA a Azure Speech y va transmitiendo
    el audio (mp3) al navegador según va llegando, en vez de esperar a que
    Azure termine de generar el clip completo antes de mandar nada. Esto es
    lo que deja que la narración arranque casi de inmediato en vez de
    quedarse "pensando" unos segundos antes del primer sonido.

    Es GET (y no POST con body) a propósito: así el <audio> del navegador
    puede pedir el audio directamente por su "src" y reproducirlo en
    streaming, en vez de tener que descargarlo completo primero con fetch.
    """
    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        raise HTTPException(
            status_code=500,
            detail="Falta configurar AZURE_SPEECH_KEY / AZURE_SPEECH_REGION."
        )

    url = f"https://{AZURE_SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    encabezados = {
        "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
    }

    cliente_http = httpx.AsyncClient(timeout=15.0)
    try:
        peticion = cliente_http.build_request(
            "POST", url, content=texto_a_ssml(texto).encode("utf-8"), headers=encabezados
        )
        respuesta = await cliente_http.send(peticion, stream=True)
    except httpx.RequestError as e:
        await cliente_http.aclose()
        raise HTTPException(status_code=502, detail=f"No se pudo contactar a Azure Speech: {e}")

    if respuesta.status_code != 200:
        cuerpo_error = await respuesta.aread()
        await respuesta.aclose()
        await cliente_http.aclose()
        print(f"Azure Speech regresó {respuesta.status_code}: {cuerpo_error[:300]}")
        raise HTTPException(status_code=502, detail="Azure Speech no pudo generar el audio.")

    async def transmitir_audio():
        try:
            async for fragmento in respuesta.aiter_bytes():
                yield fragmento
        finally:
            await respuesta.aclose()
            await cliente_http.aclose()

    return StreamingResponse(transmitir_audio(), media_type="audio/mpeg")


def _sintetizar_con_visemes(texto: str) -> tuple[bytes, list[list[int]]]:
    """
    Versión con Azure Speech SDK (no la API REST de arriba): además del
    audio, junta los eventos de "viseme" que manda Azure -offset en ms +
    id de forma de boca (0-21)- para poder mover la boca del avatar
    sílaba por sílaba de verdad, en vez de adivinar por volumen.

    Es una llamada bloqueante (el SDK usa bindings en C++, no asyncio),
    por eso el endpoint la corre en un hilo aparte con asyncio.to_thread.
    """
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = VOZ_MARIA
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
    )

    # audio_config=None: no reproduce localmente, solo nos interesan los
    # bytes en result.audio_data para mandarlos al navegador.
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    visemes: list[list[int]] = []
    synthesizer.viseme_received.connect(
        lambda evt: visemes.append([round(evt.audio_offset / 10000), evt.viseme_id])
    )

    result = synthesizer.speak_ssml_async(texto_a_ssml(texto)).get()

    if result.reason == speechsdk.ResultReason.Canceled:
        detalles = result.cancellation_details
        print(f"Azure Speech (SDK) canceló la síntesis: {detalles.reason} - {detalles.error_details}")
        raise HTTPException(status_code=502, detail="Azure Speech no pudo generar el audio.")

    return result.audio_data, visemes


@app.get("/api/tts-viseme")
async def sintetizar_voz_con_visemes(texto: str):
    """
    Prototipo: igual que /api/tts pero usando el Speech SDK en vez de la
    API REST, para obtener además los eventos de viseme y animar la boca
    del avatar sílaba por sílaba en vez de por amplitud de audio.

    A diferencia de /api/tts, este endpoint SÍ espera a que termine toda
    la síntesis antes de responder (el SDK no permite ir transmitiendo el
    audio en pedazos tan fácil como la API REST cruda), así que la
    narración tarda un poco más en arrancar. Es un trade-off aceptado para
    este prototipo. /api/tts se deja intacto sin usarse desde aquí.
    """
    if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
        raise HTTPException(
            status_code=500,
            detail="Falta configurar AZURE_SPEECH_KEY / AZURE_SPEECH_REGION."
        )

    audio_bytes, visemes = await asyncio.to_thread(_sintetizar_con_visemes, texto)

    return JSONResponse({
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
        "visemes": visemes,
    })


# Sirve la página principal (biblioteca + reproductor + análisis + chat de voz).
@app.get("/")
async def servir_interfaz():
    return FileResponse(os.path.join(RUTA_FRONTEND, "index.html"))


# Regresa el catálogo completo de videos en JSON, para que el frontend
# arme la lista de la "Biblioteca" que se ve en la página principal.
@app.get("/api/videos")
async def listar_videos():
    conexion = obtener_conexion()
    cursor = conexion.cursor()
    cursor.execute("SELECT * FROM videos")
    filas = cursor.fetchall()
    conexion.close()
    return JSONResponse(content=[dict(fila) for fila in filas])


# ==============================================================================
# --- AGREGAR VIDEOS DESDE LA PÁGINA WEB ---------------------------------------
# ==============================================================================
# Antes había que insertar cada video a mano con SQL. Con esto, se sube el
# archivo de video + su descripción desde un formulario web sencillo, y el
# servidor se encarga de: 1) guardar el archivo en videos/, y 2) crear su
# registro correspondiente en la base de datos (que es lo mismo que consulta
# obtener_catalogo_videos() para que MarIA sepa que ese video existe).

EXTENSIONES_VIDEO_PERMITIDAS = (".mp4", ".webm", ".mov")


def nombre_archivo_seguro(nombre_original: str) -> str:
    """
    Limpia el nombre del archivo antes de guardarlo en disco:
    - Quita cualquier ruta de carpeta que venga incluida (solo nos quedamos
      con el nombre del archivo), para que nadie pueda escribir fuera de
      la carpeta videos/ mandando un nombre con "../../".
    - Convierte acentos y eñes a su letra base (á->a, ñ->n, etc.) en vez de
      simplemente borrarlos, para que el nombre siga siendo legible
      ("Como_fue_el_ULTIMO_dia...mp4" en vez de "_C_mo_fue_el__LTIMO_d_a...").
    - Lo que sí sea un símbolo raro (¿, !, espacios, etc.) se reemplaza por
      guion bajo, para evitar problemas al servir el archivo por URL después.
    """
    nombre_base = os.path.basename(nombre_original)

    # NFKD separa cada letra acentuada en (letra base + acento aparte),
    # luego nos quedamos solo con la parte ASCII (la letra base).
    nombre_sin_acentos = unicodedata.normalize('NFKD', nombre_base)
    nombre_sin_acentos = nombre_sin_acentos.encode('ascii', 'ignore').decode('ascii')

    nombre_limpio = re.sub(r"[^A-Za-z0-9_.-]", "_", nombre_sin_acentos)
    return nombre_limpio


@app.post("/api/videos/agregar")
async def agregar_video(
    archivo: UploadFile = File(...),
    grado: str = Form(""),
    materia: str = Form(""),
    subtema: str = Form(""),
    duracion: int = Form(0),
    descripcion: str = Form(""),
):
    """
    Recibe el archivo de video + los datos del formulario, guarda el archivo
    físico dentro de videos/ y crea su fila en la tabla 'videos'.
    """
    if not archivo.filename.lower().endswith(EXTENSIONES_VIDEO_PERMITIDAS):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser un video (.mp4, .webm o .mov)."
        )

    nombre_final = nombre_archivo_seguro(archivo.filename)
    ruta_destino = os.path.join(RUTA_VIDEOS, nombre_final)

    # Guardamos el archivo por partes de 1 MB, en vez de cargarlo completo a
    # RAM de un jalón (importante porque los videos pueden pesar varios MB/GB).
    with open(ruta_destino, "wb") as archivo_en_disco:
        while True:
            trozo = await archivo.read(1024 * 1024)
            if not trozo:
                break
            archivo_en_disco.write(trozo)

    conexion = obtener_conexion()
    cursor = conexion.cursor()
    cursor.execute('''
        INSERT INTO videos (ruta_archivo, grado, materia, subtema, duracion, descripcion)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (nombre_final, grado, materia, subtema, duracion, descripcion))
    conexion.commit()
    conexion.close()

    return JSONResponse(content={
        "mensaje": "Video agregado correctamente.",
        "archivo": nombre_final,
    })


@app.get("/agregar")
async def servir_pagina_agregar():
    """Página con el formulario para subir videos nuevos al catálogo."""
    return FileResponse(os.path.join(RUTA_FRONTEND, "agregar.html"))
