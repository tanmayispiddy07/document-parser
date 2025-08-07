import os
from typing import List
import google.generativeai as genai
from pypdf import PdfReader
from docx import Document
import requests
import tempfile
import re

class DocumentService:
    def __init__(self):
        # Initialize Google Gemini
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        if not GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        # Configure Google Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        
        # Initialize Gemini Pro model with safety settings
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        
        self.model = genai.GenerativeModel('gemini-2.5-pro',
                                         generation_config=generation_config)

    async def download_document(self, url: str) -> str:
        """Download document from URL and save it temporarily"""
        try:
            response = requests.get(url, verify=True)
            response.raise_for_status()
            
            # Determine file type from Content-Type header or URL
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' in content_type.lower() or url.lower().endswith('.pdf'):
                suffix = '.pdf'
            elif 'word' in content_type.lower() or url.lower().endswith('.docx'):
                suffix = '.docx'
            else:
                raise ValueError("Unsupported document type. Only PDF and DOCX are supported.")

            # Create a temporary file with appropriate suffix
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.write(response.content)
            temp_file.close()  # Close the file to ensure it's written to disk
            
            # Verify the file exists and is readable
            if not os.path.exists(temp_file.name):
                raise FileNotFoundError(f"Failed to create temporary file at {temp_file.name}")
            
            return temp_file.name
            
        except requests.RequestException as e:
            raise Exception(f"Error downloading document: {str(e)}")
        except Exception as e:
            raise Exception(f"Error handling document: {str(e)}")

    async def extract_text_from_document(self, file_path: str) -> str:
        """Extract text from document based on file type"""
        try:
            if file_path.endswith(".pdf"):
                # Extract text from PDF
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            elif file_path.endswith(".docx"):
                # Extract text from DOCX
                doc = Document(file_path)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            else:
                raise ValueError("Unsupported file format")
            
            if not text.strip():
                raise ValueError("No text could be extracted from the document")
                
            return text
        except FileNotFoundError:
            raise Exception(f"Document file not found at path: {file_path}")
        except Exception as e:
            raise Exception(f"Error extracting text from document: {str(e)}")

    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into smaller chunks"""
        # Split text into sentences (simple approach)
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    async def process_document_questions(self, document_url: str, questions: List[str]) -> List[str]:
        """Process document and answer questions"""
        try:
            # Download document
            file_path = await self.download_document(document_url)
            
            # Extract text from document
            text = await self.extract_text_from_document(file_path)
            
            # Split text into chunks
            chunks = self.chunk_text(text)
            
            # Process each question
            answers = []
            for question in questions:
                # Create the prompt with the entire context
                prompt = f"""You are an expert document analyzer. Based on the following document content, provide a precise and accurate answer to the question. If the answer cannot be found in the document, say "I cannot find the answer in the provided document."

Document content:
{' '.join(chunks)}

Question: {question}

Please provide a clear, concise answer based solely on the information in the document."""
                
                # Generate answer using Gemini with retry logic and safety checks
                max_retries = 3
                retry_count = 0
                chunk_size = len(chunks) // 2  # Start with half the chunks

                while retry_count < max_retries:
                    try:
                        # Create a shorter prompt if previous attempt failed
                        if retry_count > 0:
                            chunk_size = chunk_size // 2
                            relevant_chunks = chunks[:chunk_size]
                            prompt = f"""You are an expert document analyzer. Based on the following document content, provide a precise and accurate answer to the question. If the answer cannot be found in the document, say "I cannot find the answer in the provided document."

Document content:
{' '.join(relevant_chunks)}

Question: {question}

Please provide a clear, concise answer based solely on the information in the document."""

                        response = self.model.generate_content(prompt)
                        
                        # Check if response has valid content
                        if hasattr(response, 'text') and response.text:
                            answers.append(response.text)
                            break
                        else:
                            raise ValueError("Invalid or empty response received")
                            
                    except Exception as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            answers.append(f"Could not process this question due to: {str(e)}")
                        continue

            # Cleanup temporary file
            try:
                os.unlink(file_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {file_path}: {str(e)}")
            
            return answers
            
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")
