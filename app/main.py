from fastapi import FastAPI
from pydantic import BaseModel
from ollama import AsyncClient
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)

class ChatRequest(BaseModel):
    prompt: str
    model: str = "llama3.2:3b"  # Changed to a smaller, more compatible model

@app.post("/chat")
async def chat_with_ollama(request_data: ChatRequest):
    logging.info(f"Chat request: model={request_data.model}, prompt={request_data.prompt[:50]}...")
    try:
        async_client = AsyncClient()
        response = await async_client.chat(
            model=request_data.model,
            messages=[{
                'role': 'user',
                'content': request_data.prompt
            }]
        )
        return {'response': response['message']['content']}
    except Exception as e:
        logging.error(f"Chat failed: {str(e)}")
@app.get("/health")
async def health():
    try:
        async_client = AsyncClient()
        # Quick test: list models or a minimal chat
        models = await async_client.list()
        return {"status": "ok", "models_available": len(models.get('models', []))}
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        return {"status": "error", "detail": str(e)}