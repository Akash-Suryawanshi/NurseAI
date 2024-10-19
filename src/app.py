from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from langchain.schema import HumanMessage
from pydub import AudioSegment
import speech_recognition as sr
from src.pipelines.CoT import CoTPipeline
import docx
import fitz  # PyMuPDF
import io
from src.models.base_llm import get_gpt4_llm

app = FastAPI()

cot_pipeline = CoTPipeline()

# Mount static files (CSS, JS, images) to the /static endpoint
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Define where the templates are located
templates = Jinja2Templates(directory="src/templates")

# Root endpoint to render the HTML UI
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon"}


@app.post("/process_text")
async def process_text(request: Request, query: str = Form(...)):
    global user_query
    global response
    user_query = query  # Store the submitted text
    # messages = [HumanMessage(content=query)]
    # console.log(messages) 
    diagnostic_result = cot_pipeline.generate_diagnosis(query)
    print(diagnostic_result)
    return JSONResponse({"response": diagnostic_result.get("diagnosis_result")})
    # return JSONResponse({"response": response})



# Process document files
@app.post("/process_document")
async def process_document(request: Request, file: UploadFile = File(...)):
    content = ""
    if file.filename.endswith(".docx"):
        content = extract_text_from_docx(file)
    elif file.filename.endswith(".pdf"):
        content = extract_text_from_pdf(file)
    else:
        return JSONResponse(content={"error": "Unsupported document format"}, status_code=400)
    diagnostic_result = cot_pipeline.generate_diagnosis(content)
    return JSONResponse({"response": diagnostic_result.get("diagnosis_result")})


# Process voice input
@app.post("/process_voice")
async def process_voice(request: Request, file: UploadFile = File(...)):
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
            diagnostic_result = cot_pipeline.generate_diagnosis(text)
            return JSONResponse({"response": diagnostic_result.get("diagnosis_result")})
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


