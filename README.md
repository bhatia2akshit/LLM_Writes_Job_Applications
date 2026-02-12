# LLM Writes Job Applications

## Prereqs

- Docker + Docker Compose
- `.env.local` in repo root with required API keys (e.g. `HUGGINGFACEHUB_API_TOKEN`)

## Run With Docker

```bash
docker compose up --build
```

Services:

- Backend: `http://localhost:8080`
- Frontend: `http://localhost:3000`

## Local Dev (Optional)

Backend:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```
