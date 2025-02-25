import argparse
import asyncio
import base64
import json
import os
import re
import logging
import uvicorn
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Request, WebSocketDisconnect
from fastapi.responses import JSONResponse, HTMLResponse
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
    "You are a helpful and bubbly AI assistant who loves to chat about "
    "anything the user is interested in and is prepared to offer them facts. "
    "You have a penchant for dad jokes, owl jokes, and rickrolling – subtly. "
    "Always stay positive, but work in a joke when appropriate.")
VOICE = 'alloy'

# Initialize FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Validate environment variables
if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and PHONE_NUMBER_FROM and OPENAI_API_KEY):
    raise ValueError('Missing Twilio and/or OpenAI environment variables.')

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

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
        if call_request.instructions:
            global SYSTEM_MESSAGE
            SYSTEM_MESSAGE = call_request.instructions

        call = await make_call(call_request.phone_number)
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

    async def send_to_twilio():
        nonlocal stream_sid, call_sid
        try:
            async for openai_message in openai_ws:
                response = json.loads(openai_message)
                if response['type'] == 'response.audio.delta' and response.get('delta'):
                    audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                    audio_delta = {
                        "event": "media",
                        "streamSid": stream_sid,
                        "media": {"payload": audio_payload}
                    }
                    await websocket.send_json(audio_delta)
                elif response['type'] == 'response.audio_transcript.done':
                    transcript = response.get('transcript', '')
                    await send_transcript_to_clients(call_sid, 'AI', transcript)
                elif response['type'] == 'conversation.item.input_audio_transcription.completed':
                    transcript = response.get('transcript', '')
                    await send_transcript_to_clients(call_sid, 'User', transcript)
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
            await websocket.receive_text()  # Keep connection alive
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
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": ("Greet the user with 'Hello there! I am an AI voice assistant powered by "
                         "Twilio and the OpenAI Realtime API. You can ask me for facts, jokes, or "
                         "anything you can imagine. How can I help you?'")
            }]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))

async def make_call(phone_number_to_call: str):
    """Make an outbound call using Twilio."""
    if not phone_number_to_call:
        raise ValueError("Please provide a phone number to call.")

    is_allowed = await check_number_allowed(phone_number_to_call)
    if not is_allowed:
        raise ValueError(f"The number {phone_number_to_call} is not recognized.")

    outbound_twiml = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response><Connect><Stream url="wss://{DOMAIN}/media-stream" /></Connect></Response>'
    )

    call = client.calls.create(
        from_=PHONE_NUMBER_FROM,
        to=phone_number_to_call,
        twiml=outbound_twiml
    )
    await log_call_sid(call.sid)
    return call

async def check_number_allowed(to):
    """Check if the phone number is allowed for calling."""
    try:
        incoming_numbers = client.incoming_phone_numbers.list(phone_number=to)
        if incoming_numbers:
            return True
        outgoing_caller_ids = client.outgoing_caller_ids.list(phone_number=to)
        if outgoing_caller_ids:
            return True
        return False
    except Exception as e:
        print(f"Error checking phone number: {e}")
        return False

async def log_call_sid(call_sid):
    """Log the call SID."""
    print(f"Call started with SID: {call_sid}")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Twilio AI voice assistant server.")
    parser.add_argument('--call', required=True, help="The phone number to call, e.g., '--call=+18005551212'")
    args = parser.parse_args()
    phone_number = args.call

    print('Our recommendation is to always disclose the use of AI for outbound or inbound calls.\n'
          'Reminder: All of the rules of TCPA apply even if a call is made by AI.\n'
          'Check with your counsel for legal and compliance advice.')

    loop = asyncio.get_event_loop()
    loop.run_until_complete(make_call(phone_number))

    uvicorn.run(app, host="0.0.0.0", port=PORT)