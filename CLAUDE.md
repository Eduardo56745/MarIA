# MarIA

App FastAPI ("La Teacher MarIA", prototipo para SEG Guanajuato). Entrypoint: `app.py`, frontend estático en `frontend/`, modelos de face-api.js en `models/`.

## Entorno local

Venv en `env/` (no `.venv`). Ya está creado con `requirements.txt` instalado (fastapi, uvicorn, groq, python-dotenv, python-multipart).

```powershell
.\env\Scripts\Activate.ps1
uvicorn app:app --reload
```

## Hecho en esta sesión (2026-07-13)

- Recreado `env/` (el anterior apuntaba a un Python de otra máquina, roto).
- Verificado que la app arranca bien en `localhost:8000`.
- Inicializado el repo git local y conectado a `https://github.com/Eduardo56745/MarIA`, primer push hecho.
- Renombrado `README.md.txt` → `README.md` (tenía el `.txt` de más por un guardado desde Bloc de notas).
- Creado `.claude/launch.json` para levantar el server con el tool de preview del harness.

Nada más se tocó; el código de la app sigue igual.
