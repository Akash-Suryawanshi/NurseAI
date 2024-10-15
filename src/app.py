from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment
import speech_recognition as sr
import docx
import fitz  # PyMuPDF
import io

app = FastAPI()

# Mount static files (CSS, JS, images) to the /static endpoint
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define where the templates are located
templates = Jinja2Templates(directory="templates")

# Root endpoint to render the HTML UI
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/")
async def root():
    return {"message": "Welcome to MedAssistant API"}

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon"}


# 1. Text Input Endpoint
@app.post("/process_text")
async def process_text(query: str = Form(...)):
    # Process text input (call agent here)
    result = {"input_type": "text", "content": query}
    return JSONResponse(content=result)

# 2. Document File Input (PDF or DOCX) Endpoint
@app.post("/process_document")
async def process_document(file: UploadFile = File(...)):
    content = ""
    if file.filename.endswith(".docx"):
        content = extract_text_from_docx(file)
    elif file.filename.endswith(".pdf"):
        content = extract_text_from_pdf(file)
    else:
        return JSONResponse(content={"error": "Unsupported document format"}, status_code=400)
    
    # Process document content (call agent here)
    result = {"input_type": "document", "content": content}
    return JSONResponse(content=result)

# 3. Voice Input (Audio File) Endpoint
@app.post("/process_voice")
async def process_voice(file: UploadFile = File(...)):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(file.file)

    # Convert audio to wav for processing
    wav_audio = io.BytesIO()
    audio.export(wav_audio, format="wav")
    wav_audio.seek(0)

    # Convert voice to text using SpeechRecognition
    with sr.AudioFile(wav_audio) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            result = {"input_type": "voice", "content": text}
            return JSONResponse(content=result)
        except sr.UnknownValueError:
            return JSONResponse(content={"error": "Could not understand the audio"}, status_code=400)
        except sr.RequestError:
            return JSONResponse(content={"error": "Error with the speech recognition service"}, status_code=500)

# Helper function to extract text from DOCX files
def extract_text_from_docx(file):
    doc = docx.Document(file.file)
    full_text = [paragraph.text for paragraph in doc.paragraphs]
    return '\n'.join(full_text)

# Helper function to extract text from PDF files
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.file.read(), filetype="pdf")
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text 

# To run the FastAPI server, use the command below:
# uvicorn app:app --reload
