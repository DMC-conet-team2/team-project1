from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AddRequest(BaseModel):
    a: int
    b: int

@app.post("/add")
def add(req: AddRequest):
    result = req.a + req.b
    return {"result": result}