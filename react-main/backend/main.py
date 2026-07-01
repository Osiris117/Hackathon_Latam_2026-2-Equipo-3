from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import traceback

app = FastAPI(
    title="Falcon Reservoir API",
    version="1.0.0"
)

# ==========================================================
# CORS
# ==========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Después puedes restringirlo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# IMPORTAR MÓDULOS
# ==========================================================

try:
    import genetico_v3
except Exception:
    genetico_v3 = None

try:
    import entrenar_rna
except Exception:
    entrenar_rna = None

try:
    import modelado1
except Exception:
    modelado1 = None

try:
    import qubo
except Exception:
    qubo = None

try:
    import quantum
except Exception:
    quantum = None


# ==========================================================
# HOME
# ==========================================================

@app.get("/")
def home():

    return {
        "Proyecto":"Falcon Reservoir Dashboard",
        "Estado":"Activo"
    }


# ==========================================================
# GENÉTICO
# ==========================================================

@app.get("/genetico")
def ejecutar_genetico():

    if genetico_v3 is None:

        return JSONResponse(
            status_code=500,
            content={"error":"No se pudo importar genetico_v3.py"}
        )

    try:

        resultado = genetico_v3.genetico()

        return resultado

    except Exception:

        return JSONResponse(
            status_code=500,
            content={
                "error":traceback.format_exc()
            }
        )


# ==========================================================
# RNA
# ==========================================================

@app.get("/rna")
def ejecutar_rna():

    if entrenar_rna is None:

        return JSONResponse(
            status_code=500,
            content={"error":"No se pudo importar entrenar_rna.py"}
        )

    try:

        resultado = entrenar_rna.entrenar()

        return resultado

    except Exception:

        return JSONResponse(
            status_code=500,
            content={
                "error":traceback.format_exc()
            }
        )


# ==========================================================
# MODELADO
# ==========================================================

@app.get("/modelado")
def ejecutar_modelado():

    if modelado1 is None:

        return JSONResponse(
            status_code=500,
            content={"error":"No se pudo importar modelado1.py"}
        )

    try:

        resultado = modelado1.simular()

        return resultado

    except Exception:

        return JSONResponse(
            status_code=500,
            content={
                "error":traceback.format_exc()
            }
        )


# ==========================================================
# QUBO
# ==========================================================

@app.get("/qubo")
def ejecutar_qubo():

    if qubo is None:

        return JSONResponse(
            status_code=500,
            content={"error":"No se pudo importar qubo.py"}
        )

    try:

        resultado = qubo.construir()

        return resultado

    except Exception:

        return JSONResponse(
            status_code=500,
            content={
                "error":traceback.format_exc()
            }
        )


# ==========================================================
# COMPUTACIÓN CUÁNTICA
# ==========================================================

@app.get("/quantum")
def ejecutar_quantum():

    if quantum is None:

        return JSONResponse(
            status_code=500,
            content={"error":"No se pudo importar quantum.py"}
        )

    try:

        resultado = quantum.ejecutar()

        return resultado

    except Exception:

        return JSONResponse(
            status_code=500,
            content={
                "error":traceback.format_exc()
            }
        )


# ==========================================================
# ESTADO DEL SISTEMA
# ==========================================================

@app.get("/estado")
def estado():

    return {

        "genetico":genetico_v3 is not None,

        "rna":entrenar_rna is not None,

        "modelado":modelado1 is not None,

        "qubo":qubo is not None,

        "quantum":quantum is not None

    }


# ==========================================================
# EJECUCIÓN
# ==========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
