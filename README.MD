# 1. Create and activate virtual environment
# 2. Install fastapi[standard]
```
pip install "fastapi[standard]"
```

# 3. Make a python code. On mine, its ```main.py```
```
from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
```

# 4. Use 8080 port
```
uvicorn main:app --reload --port 8080
```

# 5. Install pipx
```
pipx install mcp-proxy
```

## For the template.
```
https://apxml.com/posts/how-to-build-mcp-server-fastapi
```
## 6. Token is inside .env file

```
uv run uvicorn main:app --reload
```

# GDACS
## 1. Setup
```
pip install gdacs-api
```

# 
```
pip install sqlalchemy psycopg2-binary
```