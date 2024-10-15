from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydub import AudioSegment
import speech_recognition as sr
import docx
import fitz  # PyMuPDF
import io

app = FastAPI()

# Mount static files (CSS, JS, images) to the /static endpoint
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Define where the templates are located
templates = Jinja2Templates(directory="src/templates")

# Root endpoint to render the HTML UI
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chatbot")
async def get_chatbot(request: Request):
    global user_query

    # Use the stored user_query and check for an existing response
    response = user_query if user_query else ""

    # Render the chatbot.html template with the stored query and response
    return templates.TemplateResponse("chatbot.html", {
        "request": request,
        "content": user_query,
        "response": response  # Pass response here, even if it's empty
    })

# @app.get("/")
# async def root():
#     return {"message": "Welcome to MedAssistant API"}

@app.get("/favicon.ico")
async def favicon():
    return {"message": "No favicon"}


@app.post("/process_text")
async def process_text(request: Request, query: str = Form(...)):
    global user_query
    user_query = query  # Store the submitted text

    # Get the response from the model (for now, echo the query)
    response = query  # This should be replaced by the model's response in the future

    # Pass both the query and response to the chatbot.html page
    return templates.TemplateResponse("chatbot.html", {
        "request": request,
        "input_type": "text",
        "content": user_query,  # User's message
        "response": response  # Bot's response (echoed back for now)
    })


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
    
    return templates.TemplateResponse("chatbot.html", {"request": request, "input_type": "document", "content": content})


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
            return templates.TemplateResponse("chatbot.html", {"request": request, "input_type": "voice", "content": text})
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

# Get model response based on user input
@app.post("/ask_model")
async def ask_model(query: str = Form(...)):
    # For now, simply echo back the query as the response
    response = query
    return {"response": response}

