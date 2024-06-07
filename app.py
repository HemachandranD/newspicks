from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

from custom_agent import chain

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(app, chain, path="/helloagent", playground_type="default")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
