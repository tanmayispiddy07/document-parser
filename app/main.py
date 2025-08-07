from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.services.document_service import DocumentService
from app.models.query_models import QueryRequest, QueryResponse
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI(title="Document Query System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

document_service = DocumentService()

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # Process questions and generate answers
        answers = await document_service.process_document_questions(
            document_url=request.documents,
            questions=request.questions
        )
        
        return QueryResponse(answers=answers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
