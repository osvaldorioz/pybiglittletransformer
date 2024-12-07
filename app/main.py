from fastapi import FastAPI
import numpy as np
from big_little_transformer import BigLittleTransformer
import time
from pydantic import BaseModel
from typing import List
import json
import random

def listaAleatorios(n: int):
      lista = [0]  * n
      for i in range(n):
          lista[i] = random.randint(0, 10000)* 0.0001
      return lista

app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]

@app.post("/biglittle")
async def calculo(input_dim: int, output_dim: int):
    start = time.time()

    # Crear entrada aleatoria (batch_size=4, input_dim=8)
    input_data = np.random.rand(output_dim, input_dim).astype(np.float64)

    # Inicializar el modelo
    model = BigLittleTransformer(input_dim, output_dim)

    # Inferencia
    output = model.forward(input_data)
    #print("Output del modelo:", output)

    end = time.time()

    var1 = end - start

    j1 = {
        "Time taken in seconds": var1,
        "Data": input_data,
        "Output del modelo": output
    }
    jj = json.dumps(str(j1))

    return jj