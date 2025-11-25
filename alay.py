import asyncio
from datetime import datetime, timedelta
import os
import re
import json
from typing import Dict, Any

import httpx
import requests
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi_mcp import FastApiMCP
from geopy.distance import geodesic

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Apxml Awesome MCP",
    description="API services for internal use.",
)

# --- Models ---
class User(BaseModel):
    userId: str
    latitude_loc: float
    longitude_loc: float
    is_unsafe: bool

class AIReasonRequest(BaseModel):
    userId: str

class Rescuer(BaseModel):
    userId: str
    latitude_loc: float
    longitude_loc: float
    is_available: bool

class RescueTask(BaseModel):
    taskId: str
    unsafe_user: str
    rescuer_assigned: str
    hazard_level: str
    location: Dict[str, float]
    created_at: str

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_json(self, websocket: WebSocket, message: Dict[str, Any]):
        await websocket.send_json(message)

# --- Health check ---
@app.get("/")
async def root():
    return {"message": "Server is running"}

# --- API helpers ---
async def fetch_unsafe_users_api() -> list[Dict[str, Any]]:
    """"
    url = "https://your-backend.example.com/api/unsafe_users"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
    results = []
    for u in data:
        /**
        results.append({
            "userId": u["userId"],
            "latitude_loc": u["latitude_loc"],
            "longitude_loc": u["longitude_loc"],
            "is_unsafe": u.get("is_unsafe", True)
        })
    return results
    """""
    # Mocked data for demonstration
    return [
        {"userId": "user123", "latitude_loc": 14.5995, "longitude_loc": 120.9842, "is_unsafe": True},
    ]

async def fetch_available_rescuers_api() -> list[Dict[str, Any]]:
    """
    url = "https://your-backend.example.com/api/available_rescuers"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
    return data
    """
    # Mocked data for demonstration
    return [
        {"userId": "rescuer456", "latitude_loc": 14.5990, "longitude_loc": 120.9820, "is_available": True},
    ]
# --- Weather API ---
async def get_weather_data(lat: float, lon: float):
    API_KEY = os.getenv("WEATHER_API_KEY")
    if not API_KEY:
        raise HTTPException(status_code=500, detail="❌ Missing WEATHER_API_KEY in .env")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        weather_desc = data["weather"][0]["main"].lower()
        wind_speed = data["wind"]["speed"]
        severe = False
        reason = ""

        if "storm" in weather_desc or "typhoon" in weather_desc:
            severe = True
            reason = "Severe storm/typhoon detected"
        elif data.get("rain", {}).get("1h", 0) > 20:
            severe = True
            reason = "Heavy rainfall (>20mm/h)"

        return {
            "coordinates": {"lat": lat, "lon": lon},
            "condition": weather_desc,
            "temperature_c": data["main"]["temp"],
            "wind_speed_mps": wind_speed,
            "severe": severe,
            "reason": reason
        }

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Weather API request failed: {e}")
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Error processing weather data: {e}")

# --- News API ---
async def fetch_local_news(city_name: str, limit: int = 5):
    API_KEY = os.getenv("NEWS_API_KEY")
    if not API_KEY:
        return []

    query = f"{city_name} Philippines"
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&language=en&pageSize={limit}&apiKey={API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        articles = []
        for a in data.get("articles", []):
            articles.append({
                "title": a.get("title"),
                "description": a.get("description"),
                "url": a.get("url"),
                "publishedAt": a.get("publishedAt")
            })
        return articles

    except Exception as e:
        print(f"[News] Failed to fetch news: {e}")
        return []

# --- Earthquake API ---
async def get_earthquake_data(lat: float, lon: float, radius_km: int = 1000):
    min_lat, max_lat = 4.6, 21.1
    min_lon, max_lon = 116.9, 126.6
    start_time = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
    url = (
        f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
        f"format=geojson&starttime={start_time}"
        f"&minlatitude={min_lat}&maxlatitude={max_lat}"
        f"&minlongitude={min_lon}&maxlongitude={max_lon}"
    )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

        earthquakes = []
        for feature in data.get("features", []):
            props = feature["properties"]
            coords = feature["geometry"]["coordinates"]
            quake_lat, quake_lon = coords[1], coords[0]
            distance = geodesic((lat, lon), (quake_lat, quake_lon)).km
            if distance <= radius_km:
                earthquakes.append({
                    "magnitude": props.get("mag"),
                    "place": props.get("place"),
                    "time_utc": datetime.utcfromtimestamp(props["time"] / 1000).isoformat() + "Z",
                    "distance_km": round(distance, 2),
                    "url": props.get("url")
                })

        return {
            "location": {"latitude": lat, "longitude": lon, "radius_km": radius_km},
            "earthquakes": earthquakes,
            "count": len(earthquakes)
        }

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Earthquake API request failed: {e}")
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Error processing earthquake data: {e}")

# --- Hazard Check ---
@app.get("/hazard_check", operation_id="hazard_check")
async def hazard_check():
    unsafe_users = await fetch_unsafe_users_api()
    if not unsafe_users:
        return {"message": "No unsafe users currently."}

    hazards = []
    for user in unsafe_users:
        lat, lon = user["latitude_loc"], user["longitude_loc"]
        weather = await get_weather_data(lat, lon)
        earthquake = await get_earthquake_data(lat, lon)
        hazards.append({
            "user_id": user["userId"],
            "location": {"lat": lat, "lon": lon},
            "hazards": {
                "weather": weather,
                "earthquakes": earthquake["earthquakes"]
            }
        })

    return {"hazard_checks": hazards}

# --- AI Reasoning ---
@app.post("/ai/reason", operation_id="ai_reason")
async def ai_reason(request: AIReasonRequest):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="❌ Missing OPENAI_API_KEY in .env")
    
    unsafe_users = await fetch_unsafe_users_api()
    user = next((u for u in unsafe_users if u["userId"] == request.userId), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    lat, lon = user["latitude_loc"], user["longitude_loc"]

    weather = await get_weather_data(lat, lon)
    earthquake = await get_earthquake_data(lat, lon)
    local_news = await fetch_local_news(lat, lon)

    prompt = f"""
You are a hazard-analysis assistant. Analyze the following user, hazard, and local news data,
then respond with a single JSON object ONLY with keys: risk_level, summary, recommended_action.

USER DATA:
{json.dumps(user)}

WEATHER DATA:
{json.dumps(weather)}

EARTHQUAKE DATA:
{json.dumps(earthquake)}

LOCAL NEWS:
{json.dumps(local_news)}
"""

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4.1-mini",
        "messages": [
            {"role": "system", "content": "You are a precise hazard assessment AI. Reply ONLY with JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 800
    }

    try:
        res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        ai_text = data["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API call failed: {e}")

    # Parse JSON safely
    cleaned = re.sub(r"```(?:json)?\n?", "", ai_text).replace("```", "")
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    json_str = match.group(0) if match else cleaned.strip()

    try:
        ai_json = json.loads(json_str)
    except Exception:
        ai_json = {
            "risk_level": "High",
            "summary": "Simulated high risk near user location",
            "recommended_action": "Dispatch nearest rescuer immediately"
        }

    required = {"risk_level", "summary", "recommended_action"}
    if not required.issubset(ai_json.keys()):
        raise HTTPException(status_code=500, detail=f"AI JSON missing required keys: {list(ai_json.keys())}")

    return {
        "user": user,
        "hazard_raw": {"weather": weather, "earthquake": earthquake, "news": local_news},
        "analysis": ai_json
    }


# --- AI Reason + Rescue ---
@app.post("/ai/reason_and_rescue", operation_id="ai_reason_and_rescue")
async def ai_reason_and_rescue(request: AIReasonRequest):
    unsafe_users = await fetch_unsafe_users_api()
    user = next((u for u in unsafe_users if u["userId"] == request.userId), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    lat, lon = user["latitude_loc"], user["longitude_loc"]
    weather = await get_weather_data(lat, lon)
    earthquake = await get_earthquake_data(lat, lon)
    ai_response = await ai_reason(request)
    risk_level = ai_response["analysis"]["risk_level"].lower()

    if risk_level != "high":
        return {"user": user, "hazard_raw": {"weather": weather, "earthquake": earthquake}, "analysis": ai_response["analysis"], "rescue_created": False}

    rescuers = await fetch_available_rescuers_api()
    nearest_rescuer = None
    nearest_dist = float("inf")
    for r in rescuers:
        dist = geodesic((lat, lon), (r["latitude_loc"], r["longitude_loc"])).km
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_rescuer = r

    if not nearest_rescuer:
        return {"user": user, "hazard_raw": {"weather": weather, "earthquake": earthquake}, "analysis": ai_response["analysis"], "rescue_created": False, "message": "No available rescuer nearby."}

    new_task = {
        "taskId": f"task-{datetime.utcnow().timestamp()}",
        "unsafe_user": user["userId"],
        "rescuer_assigned": nearest_rescuer["userId"],
        "hazard_level": risk_level,
        "location": {"lat": lat, "lon": lon},
        "created_at": datetime.utcnow().isoformat() + "Z"
    }

    return {
        "user": user,
        "hazard_raw": {"weather": weather, "earthquake": earthquake},
        "analysis": ai_response["analysis"],
        "rescue_created": True,
        "task": new_task,
        "rescuer_assigned": nearest_rescuer["userId"]
    }
# --- WebSocket Manager ---
manager = ConnectionManager()

# ----------------- Agent Loop -----------------
async def agent_loop():
    while True:
        print("\n===== AGENTIC AI CYCLE START =====")
        unsafe_users = await fetch_unsafe_users_api()
        if not unsafe_users:
            print("[AI] No unsafe users found.")
        else:
            print(f"[AI] Found {len(unsafe_users)} unsafe users")

        for user in unsafe_users:
            user_id = user["userId"]
            print(f"\n[AI] Processing user {user_id}")

            try:
                response = await ai_reason_and_rescue(AIReasonRequest(userId=user_id))
                analysis = response["analysis"]
                rescue_created = response.get("rescue_created", False)
                print(f"[AI] Risk Level: {analysis['risk_level']}")
                if rescue_created:
                    print(f"[AI] Rescue Task Created → Rescuer: {response['rescuer_assigned']} Task ID: {response['task']['taskId']}")
                else:
                    print("[AI] No rescue task created.")

                # Notify WebSocket clients
                for ws in manager.active_connections:
                    await manager.send_json(ws, {
                        "userId": user_id,
                        "analysis": analysis,
                        "rescue_created": rescue_created,
                        "task": response.get("task")
                    })

            except Exception as e:
                print(f"[AI] Error processing user {user_id}: {e}")

        print("===== AGENTIC AI CYCLE END – Waiting 60s =====\n")
        await asyncio.sleep(60)


# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            user_id = data.get("userId")

            if not user_id:
                await manager.send_json(websocket, {"error": "Missing userId"})
                continue

            try:
                # Use your AI Reason + Rescue logic
                response = await ai_reason_and_rescue(AIReasonRequest(userId=user_id))
                await manager.send_json(websocket, {"status": "success", "result": response})

            except HTTPException as e:
                await manager.send_json(websocket, {"status": "error", "detail": e.detail})
            except Exception as e:
                await manager.send_json(websocket, {"status": "error", "detail": str(e)})

    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.on_event("startup")
async def start_agent_loop_event():
    print("🚀 Starting Agent Loop...")
    asyncio.create_task(agent_loop())

# --- MCP Integration ---
mcp = FastApiMCP(
    app,
    name="Apxml API Services",
    description="Tools for disaster risk response.",
    describe_all_responses=True,
    describe_full_response_schema=True,
    include_operations=[
        "hazard_check",
        "ai_reason",
        "ai_reason_and_rescue",
    ],
)
mcp.mount(mount_path="/disaster_mcp")
mcp.setup_server()
