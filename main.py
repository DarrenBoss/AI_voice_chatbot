import asyncio
import base64
import json
import logging
import os
import re

import uvicorn
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from twilio.rest import Client
from websockets.client import connect

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# Configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
PHONE_NUMBER_FROM = os.getenv('PHONE_NUMBER_FROM')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
raw_domain = os.getenv('DOMAIN', '')
DOMAIN = re.sub(r'(^\w+:|^)\/\/|\/+$', '', raw_domain)  # Strip protocols and trailing slashes
PORT = int(os.getenv('PORT', 8000))
SYSTEM_MESSAGE = (
    "You are a helpful AI assistant who loves to chat about "
    "anything the user is interested in and is prepared to offer them facts. "
    "You are cynical and love black humour."
    "Be very attentive to the person you are speaking to and always give them a chance to jump in")
VOICE = 'alloy'

# Initialize FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Validate environment variables
if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and PHONE_NUMBER_FROM and OPENAI_API_KEY):
    raise ValueError('Missing Twilio and/or OpenAI environment variables.')

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
logging.getLogger('twilio').setLevel(logging.DEBUG)

# Dictionary to track transcript clients by call_sid
transcript_clients = {}  # {call_sid: [WebSocket, ...]}

# Pydantic model for call request
class CallRequest(BaseModel):
    phone_number: str
    instructions: str = ""

# Routes
@app.get('/', response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/make_call')
async def initiate_call(call_request: CallRequest):
    """Initiate a call and return the call_sid."""
    try:
        # Store instructions but don't modify SYSTEM_MESSAGE
        instructions = call_request.instructions

        call = await make_call(call_request.phone_number, instructions)
        return {
            "message": f"Call initiated to {call_request.phone_number}",
            "call_sid": call.sid
        }
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.websocket('/media-stream')
async def handle_media_stream(websocket: WebSocket):
    """Handle media stream from Twilio and process OpenAI responses."""
    await websocket.accept()
    stream_sid = None
    call_sid = None

    async def receive_from_twilio():
        nonlocal stream_sid, call_sid
        try:
            async for message in websocket.iter_text():
                data = json.loads(message)
                if data['event'] == 'start':
                    stream_sid = data['start']['streamSid']
                    call_sid = data['start']['callSid']  # Extract call_sid
                    print(f"Incoming stream started {stream_sid} for call {call_sid}")
                    
                    # Store call_sid for later retrieval in send_initial_conversation_item
                    # This ensures we can access the correct instructions
                    if call_sid not in transcript_clients:
                        transcript_clients[call_sid] = []
                        
                elif data['event'] == 'media' and openai_ws.open:
                    audio_append = {
                        "type": "input_audio_buffer.append",
                        "audio": data['media']['payload']
                    }
                    await openai_ws.send(json.dumps(audio_append))
        except WebSocketDisconnect:
            print("Twilio WebSocket disconnected")
            if openai_ws.open:
                await openai_ws.close()

    async def send_to_twilio():  # All messages coming from OpenAI
        nonlocal stream_sid, call_sid
        # Initialize variables to track the assistant's message and audio duration
        last_assistant_item_id = None
        current_audio_duration_ms = 0
        try:
            async for openai_message in openai_ws:
                response = json.loads(openai_message)
                logger.info(response)
                if response["type"] == "response.audio.delta" and response.get("delta"):
                    audio_chunk = base64.b64decode(response["delta"])
                    audio_payload = base64.b64encode(audio_chunk).decode("utf-8")
                    audio_delta = {
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": audio_payload}
                    }
                    await websocket.send_json(audio_delta)

                elif response["type"] == "input_audio_buffer.speech_started":
                    # Cancel the current response cleanly
                    logger.info("Cancelling current response")
                    cancel_message = {"type": "response.cancel"}
                    await openai_ws.send(json.dumps(cancel_message))
                    # Rely on server-side turn detection; no manual truncation

                elif response["type"] == "response.audio_transcript.done":
                    logger.info("Transcript received")
                    transcript = response.get("transcript", "")
                    await send_transcript_to_clients(call_sid, "AI", transcript)

                elif response["type"] == "conversation.item.input_audio_transcription.completed":
                    logger.info("transcript received")
                    transcript = response.get("transcript", "")
                    await send_transcript_to_clients(call_sid, "User", transcript)

        except websockets.exceptions.ConnectionClosed:
            print("OpenAI WebSocket connection closed")
        except Exception as e:
            print(f"Error in send_to_twilio: {e}")

    # Connect to OpenAI Realtime API
    async with connect(
        'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview',
        extra_headers=[
            ('Authorization', f'Bearer {OPENAI_API_KEY}'),
            ('OpenAI-Beta', 'realtime=v1')
        ]
    ) as openai_ws:
        await initialize_session(openai_ws)
        await asyncio.gather(receive_from_twilio(), send_to_twilio())

@app.websocket('/transcript-stream')
async def transcript_stream(websocket: WebSocket):
    """Stream transcripts to the frontend based on call_sid."""
    await websocket.accept()
    call_sid = websocket.query_params.get('call_sid')
    if not call_sid:
        await websocket.close(code=1008, reason="Missing call_sid")
        return

    if call_sid not in transcript_clients:
        transcript_clients[call_sid] = []
    transcript_clients[call_sid].append(websocket)

    try:
        while True:
            await websocket.receive_text()  # Keep connection alive fff
    except WebSocketDisconnect:
        transcript_clients[call_sid].remove(websocket)
        if not transcript_clients[call_sid]:
            del transcript_clients[call_sid]

# Helper Functions
async def send_transcript_to_clients(call_sid, speaker, transcript):
    """Send transcript to all connected clients for a given call_sid."""
    if call_sid in transcript_clients:
        for client in transcript_clients[call_sid]:
            try:
                await client.send_json({
                    'speaker': speaker,
                    'transcript': transcript
                })
            except Exception as e:
                print(f"Error sending transcript: {e}")

async def initialize_session(openai_ws):
    """Initialize the OpenAI session with settings and initial greeting."""
    session_update = {
        "type": "session.update",
        "session": {
            "turn_detection": {"type": "server_vad"},
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_ulaw",
            "input_audio_transcription": {"model": "whisper-1"},
            "voice": VOICE,
            "instructions": SYSTEM_MESSAGE,
            "modalities": ["text", "audio"],
            "temperature": 0.8,
        }
    }
    await openai_ws.send(json.dumps(session_update))
    await send_initial_conversation_item(openai_ws)

async def send_initial_conversation_item(openai_ws):
    """Send the initial greeting message to OpenAI."""
    # Get call SID from stream SID (which should be available in the WebSocket context)
    call_sid = None
    for sid in transcript_clients:
        if isinstance(sid, str) and sid.endswith("_instructions"):
            call_sid = sid.replace("_instructions", "")
            break
    
    # Default message
    default_message = ("Greet the user with 'Hello there! I am Darjan's personal assistant "
                      "Darjan would like to know how is it going for you at the gym." 
                      "Answer here and I will convey the messsage to him.'")
    
    # Get custom instructions if available
    custom_instructions = ""
    if call_sid and call_sid + "_instructions" in transcript_clients:
        custom_instructions = transcript_clients[call_sid + "_instructions"]
    
    # Use custom instructions if provided, otherwise use default
    message_text = custom_instructions if custom_instructions else default_message
    
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": message_text
            }]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))

async def make_call(phone_number_to_call: str, instructions: str = ""):
    """Make an outbound call using Twilio."""
    if not phone_number_to_call:
        raise ValueError("Please provide a phone number to call.")

    outbound_twiml = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response><Connect><Stream url="wss://{DOMAIN}/media-stream" /></Connect></Response>'
    )

    call = client.calls.create(
        from_=PHONE_NUMBER_FROM,
        to=phone_number_to_call,
        twiml=outbound_twiml
    )
    
    # Store instructions in transcript_clients for use in send_initial_conversation_item
    if call.sid not in transcript_clients:
        transcript_clients[call.sid] = []
    transcript_clients[call.sid + "_instructions"] = instructions
    
    await log_call_sid(call.sid)
    return call

async def log_call_sid(call_sid):
    """Log the call SID."""
    print(f"Call started with SID: {call_sid}")

# Main execution
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)