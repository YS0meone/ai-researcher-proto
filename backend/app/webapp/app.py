from fastapi import FastAPI

app = FastAPI()


@app.post("/ingest")
async def ingest_papers():
    return {"Hello": "World"}
