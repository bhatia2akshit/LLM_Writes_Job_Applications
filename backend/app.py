from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from agent import call_agent
from helper import extract_pdf_text, reset_token_usage, print_token_usage
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache for the last uploaded CV text.
_CACHED_CV_TEXT: Optional[str] = None
    

@app.get("/health")
def read_root():
    print("api: health check")
    return {"we are healthy"}

@app.post("/upload/")
def upload_and_submit(
    jd: str = Form(...),
    file: Optional[UploadFile] = File(None),
):
    global _CACHED_CV_TEXT
    print("api: /upload start")
    reset_token_usage()
    if file is not None:
        print("api: extracting CV text")
        # UploadFile returns bytes, decode to text before passing to the agent.
        _CACHED_CV_TEXT = extract_pdf_text(file.file)
        print("api: CV text cached")

    if not _CACHED_CV_TEXT:
        print("api: error - no CV cached")
        raise HTTPException(status_code=400, detail="No CV uploaded yet.")

    print('calling the agent.')
    result_text = call_agent(_CACHED_CV_TEXT, jd)
    print('agent calling is finished.')
    print_token_usage()
    print("api: /upload complete")
    return result_text

@app.get("/")
def index():

    print("api: index")
    return {"Hello": "World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
