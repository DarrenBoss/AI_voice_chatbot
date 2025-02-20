import argparse
import asyncio
import base64
import json
import os
import re

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel
from twilio.rest import Client
from websockets.client import connect

# Load environment variables
load_dotenv()

# Configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
PHONE_NUMBER_FROM = os.getenv('PHONE_NUMBER_FROM')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
raw_domain = os.getenv('DOMAIN', '')
DOMAIN = re.sub(r'(^\w+:|^)\/\/|\/+$', '', raw_domain)  # Strip protocols and trailing slashes from DOMAIN

PORT = int(os.getenv('PORT', 8000))
SYSTEM_MESSAGE = (
    "You are a helpful and bubbly AI assistant who loves to chat about "
    "anything the user is interested in and is prepared to offer them facts. "
    "You have a penchant for dad jokes, owl jokes, and rickrolling – subtly. "
    "Always stay positive, but work in a joke when appropriate."
)
VOICE = 'alloy'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated', 'response.done',
    'input_audio_buffer.committed', 'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started', 'session.created'
]

# Initialize FastAPI and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Validate environment variables
if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and PHONE_NUMBER_FROM and OPENAI_API_KEY):
    raise ValueError(
        'Missing Twilio and/or OpenAI environment variables. Please set them in the .env file.'
    )

# Initialize Twilio client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Store active WebSocket connections for media streams and transcripts
active_connections = []  # For Twilio media streams
transcript_connections = []  # For transcript clients

# Call request model
class CallRequest(BaseModel):
    phone_number: str
    instructions: str = ""

# Index page endpoint
@app.get('/', response_class=HTMLResponse)
async def index_page(request: Request):
    """Serve the index.html template."""
    return templates.TemplateResponse("index.html", {"request": request})

# Initiate call endpoint
@app.post('/make_call')
async def initiate_call(call_request: CallRequest):
    """Initiate an outbound call with optional custom instructions."""
    try:
        if call_request.instructions:
            global SYSTEM_MESSAGE
            SYSTEM_MESSAGE = call_request.instructions

        await make_call(call_request.phone_number)
        return {"message": f"Call initiated to {call_request.phone_number}"}
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Media stream WebSocket endpoint
@app.websocket('/media-stream')
async def handle_media_stream(websocket: WebSocket):
    """Handle WebSocket connections between Twilio and OpenAI with reconnection logic."""
    await websocket.accept()
    active_connections.append(websocket)
    print(f"Client connected. Total connections: {len(active_connections)}")

    reconnect_attempts = 0
    max_reconnect_attempts = 5  # Maximum number of reconnection attempts
    delay = 1  # Initial delay in seconds

    while True:
        try:
            async with connect(
                'wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview',
                extra_headers=[
                    ('Authorization', f'Bearer {OPENAI_API_KEY}'),
                    ('OpenAI-Beta', 'realtime=v1')
                ]
            ) as openai_ws:
                print("Connected to OpenAI WebSocket.")
                # Reset retry counter and backoff delay on successful connection
                reconnect_attempts = 0
                delay = 1

                # Initialize the session on OpenAI WebSocket
                await initialize_session(openai_ws)
                stream_sid = None

                async def receive_from_twilio():
                    nonlocal stream_sid
                    try:
                        async for message in websocket.iter_text():
                            data = json.loads(message)
                            print(f"[receive_from_twilio]: {data['event']}")
                            if data['event'] == 'media' and openai_ws.open:
                                audio_append = {
                                    "type": "input_audio_buffer.append",
                                    "audio": data['media']['payload']
                                }
                                await openai_ws.send(json.dumps(audio_append))
                            elif data['event'] == 'start':
                                stream_sid = data['start']['streamSid']
                    except WebSocketDisconnect:
                        print("[receive_from_twilio] Twilio client disconnected.")
                        # Terminate reconnection loop if the client disconnects
                        raise

                async def send_to_twilio():
                    nonlocal stream_sid
                    try:
                        async for openai_message in openai_ws:
                            try:
                                response = json.loads(openai_message)
                                print(f"[send_to_twilio]: {response['type']}")

                                # Handle AI transcripts
                                if response['type'] == 'response.audio_transcript.delta' and 'delta' in response:
                                    await broadcast_transcript(f"AI: {response['delta']}")
                                # Handle user transcripts
                                elif response['type'] == 'conversation.item.created':
                                    item = response.get('item', {})
                                    if item.get('type') == 'message' and item.get('role') == 'user':
                                        for content in item.get('content', []):
                                            if content.get('type') == 'input_text':
                                                await broadcast_transcript(f"User: {content['text']}")
                                # Handle audio responses to Twilio
                                elif response['type'] == 'response.audio.delta' and 'delta' in response:
                                    try:
                                        audio_payload = base64.b64encode(
                                            base64.b64decode(response['delta'])
                                        ).decode('utf-8')
                                        audio_delta = {
                                            "event": "media",
                                            "streamSid": stream_sid,
                                            "media": {
                                                "payload": audio_payload
                                            }
                                        }
                                        await websocket.send_json(audio_delta)
                                    except Exception as e:
                                        print(f"Error processing audio data: {e}")
                            except Exception as e:
                                print(f"Error processing OpenAI message: {e}")
                    except Exception as e:
                        print(f"Error in send_to_twilio: {e}")

                # Run both receivers concurrently
                await asyncio.gather(receive_from_twilio(), send_to_twilio())

        except WebSocketDisconnect:
            # Client (Twilio) closed the connection; exit the loop
            print("Twilio client disconnected. Closing reconnection loop.")
            break
        except Exception as e:
            # Handle disconnects from the OpenAI WebSocket
            reconnect_attempts += 1
            if reconnect_attempts > max_reconnect_attempts:
                print("Maximum reconnection attempts reached. Giving up.")
                break
            print(f"Error with OpenAI WebSocket: {e}. Reconnecting in {delay} seconds (attempt {reconnect_attempts}/{max_reconnect_attempts}).")
            await asyncio.sleep(delay)
            # Exponential backoff up to a ceiling (e.g., 30 seconds)
            delay = min(delay * 2, 30)
            continue

    # Clean up connection from active connections list
    if websocket in active_connections:
        active_connections.remove(websocket)

# Transcript stream WebSocket endpoint
@app.websocket('/transcript-stream')
async def handle_transcript_stream(websocket: WebSocket):
    """Handle WebSocket connections for real-time transcript updates."""
    await websocket.accept()
    transcript_connections.append(websocket)
    print(f"Transcript client connected. Total connections: {len(transcript_connections)}")
    try:
        while True:
            await websocket.receive_text()  # Keep the connection alive
    except WebSocketDisconnect:
        print(f"Transcript client disconnected.")
    except Exception as e:
        print(f"Error with transcript client: {e}")
    finally:
        if websocket in transcript_connections:
            transcript_connections.remove(websocket)
        print(f"Transcript client removed. Total connections: {len(transcript_connections)}")

# Initialize session with OpenAI
async def send_initial_conversation_item(openai_ws):
    """Send initial conversation so AI talks first."""
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_text",
                "text": "Greet the user with 'Hello there! I am Darjans personal assistant. How can I help you today?'"
            }]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))

async def initialize_session(openai_ws):
    """Control initial session with OpenAI."""
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
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    # Have the AI speak first
    await send_initial_conversation_item(openai_ws)

# Utility functions
async def check_number_allowed(to):
    """Check if a number is allowed to be called."""
    try:
        # Uncomment these lines to test numbers. Only add numbers you have permission to call
        # OVERRIDE_NUMBERS = ['+18005551212']
        # if to in OVERRIDE_NUMBERS:
        #     return True

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

async def make_call(phone_number_to_call: str):
    """Make an outbound call."""
    if not phone_number_to_call:
        raise ValueError("Please provide a phone number to call.")

    is_allowed = await check_number_allowed(phone_number_to_call)
    if not is_allowed:
        raise ValueError(
            f"The number {phone_number_to_call} is not recognized as a valid outgoing number or caller ID."
        )

    # Ensure compliance with applicable laws and regulations
    # All of the rules of TCPA apply even if a call is made by AI.
    # Do your own diligence for compliance.

    outbound_twiml = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response><Connect><Stream url="wss://{DOMAIN}/media-stream" /></Connect></Response>'
    )

    call = client.calls.create(from_=PHONE_NUMBER_FROM, to=phone_number_to_call, twiml=outbound_twiml)

    await log_call_sid(call.sid)

async def log_call_sid(call_sid):
    """Log the call SID."""
    print(f"Call started with SID: {call_sid}")

async def broadcast_transcript(message):
    """Broadcast the transcript to all connected transcript clients."""
    data = {
        "type": "transcript",
        "text": message
    }
    for connection in transcript_connections.copy():
        try:
            await connection.send_text(json.dumps(data))
            print(f"Sent transcript: {message}")
        except Exception as e:
            print(f"Error sending to transcript client: {e}")
            if connection in transcript_connections:
                transcript_connections.remove(connection)

# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Twilio AI voice assistant server."
    )
    parser.add_argument(
        '--call',
        required=True,
        help="The phone number to call, e.g., '--call=+18005551212'"
    )
    args = parser.parse_args()

    phone_number = args.call

    print(
        'Our recommendation is to always disclose the use of AI for outbound or inbound calls.\n'
        'Reminder: All of the rules of TCPA apply even if a call is made by AI.\n'
        'Check with your counsel for legal and compliance advice.'
    )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(make_call(phone_number))

    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=True)