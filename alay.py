from datetime import datetime, timedelta
import os
from fastapi import FastAPI, Depends, HTTPException
import httpx
import requests
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi_mcp import FastApiMCP, AuthConfig
from dotenv import load_dotenv
from geopy.distance import geodesic  # pip install geopy
import json
import re
from typing import Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from models import User, RescuerStatus, UserStatusUpdate
from database import get_db
from fastapi import Depends

# Load environment variables from the .env file
load_dotenv()

app = FastAPI(
    title="Apxml Awesome MCP",
    description="API services for internal use.",
)

@app.get("/")
async def root():
    return {"message": "Server is running"}

# Tool endpoints
class User(BaseModel):
    userId: str
    latitude_loc: float
    longitude_loc: float
    is_unsafe:bool

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
    

@app.get(
    "/users/unsafe",
    operation_id="get_unsafe_users",
    description="Fetch all users whose latest status update is 'Unsafe'.",
    tags=["users"],
    response_model=list[User],  # <-- list
)
def get_unsafe_users(db: Session = Depends(get_db)):
    # Subquery: latest update datetime per user
    latest_updates_subq = (
        db.query(
            UserStatusUpdate.user_id,
            func.max(UserStatusUpdate.update_datetime).label("latest_update")
        )
        .group_by(UserStatusUpdate.user_id)
        .subquery()
    )

    # Join back to get latest status
    latest_status_subq = (
        db.query(UserStatusUpdate)
        .join(
            latest_updates_subq,
            (UserStatusUpdate.user_id == latest_updates_subq.c.user_id) &
            (UserStatusUpdate.update_datetime == latest_updates_subq.c.latest_update)
        )
        .subquery()
    )

    # Join with users and filter for Unsafe
    unsafe_users = (
        db.query(User)
        .join(latest_status_subq, User.id == latest_status_subq.c.user_id)
        .filter(latest_status_subq.c.status == 'Unsafe')
        .all()
    )

    return unsafe_users

async def get_unsafe_users_with_coords(db: Session):
    """
    Return all unsafe users with latest coordinates (lat/lon extracted from Google Maps URL)
    """
    latest_updates_subq = (
        db.query(
            UserStatusUpdate.user_id,
            func.max(UserStatusUpdate.update_datetime).label("latest_update")
        )
        .group_by(UserStatusUpdate.user_id)
        .subquery()
    )

    latest_status_subq = (
        db.query(UserStatusUpdate)
        .join(
            latest_updates_subq,
            (UserStatusUpdate.user_id == latest_updates_subq.c.user_id) &
            (UserStatusUpdate.update_datetime == latest_updates_subq.c.latest_update)
        )
        .subquery()
    )

    unsafe_users = (
        db.query(User, latest_status_subq.c.location, latest_status_subq.c.status)
        .join(latest_status_subq, User.id == latest_status_subq.c.user_id)
        .filter(latest_status_subq.c.status == "Unsafe")
        .all()
    )

    results = []
    for user, location, status in unsafe_users:
        if not location:
            continue

        # Extract lat/lon from Google Maps URL
        match = re.search(r"@?(-?\d+\.\d+),(-?\d+\.\d+)", location)
        if not match:
            continue
        lat, lon = float(match.group(1)), float(match.group(2))
        results.append({
            "user": user,
            "lat": lat,
            "lon": lon
        })
    return results

@app.get(   
    "/weather",
    operation_id="get_weather_data",
    description="Fetches weather data for given coordinates.",
    tags=["weather"],
)
# check weather data 
async def get_weather_data(lat: float, lon: float):
    """
    Fetch current weather and check for severe conditions like typhoon/heavy rain.
    """
    API_KEY = os.getenv("WEATHER_API_KEY")
    if not API_KEY:
        raise HTTPException(status_code=500, detail="❌ Missing API key. Set WEATHER_API_KEY in .env file.")

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Current weather
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            weather_desc = data["weather"][0]["main"].lower()  # e.g., "rain", "storm", "clear"
            wind_speed = data["wind"]["speed"]

            # Check if severe
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


# This will get the user location and check for nearby earthquakes
@app.get(
    "/earthquake",
    operation_id="get_earthquake_data",
    description="Fetches earthquake alert and coordination data for given coordinates.",
    tags=["earthquakes"],
)

async def get_earthquake_data(lat: float, lon: float, radius_km: int = 1000):
    """
    Fetches earthquakes near a given coordinate within radius_km (default 1000 km).
    Only considers earthquakes in the Philippines in the last 24 hours.
    """
    latitude = lat
    longitude = lon

    # Philippines bounding box
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
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

        earthquakes = []
        for feature in data.get("features", []):
            props = feature["properties"]
            coords = feature["geometry"]["coordinates"]
            quake_lat, quake_lon = coords[1], coords[0]

            # Distance from user
            distance = geodesic((latitude, longitude), (quake_lat, quake_lon)).km
            if distance <= radius_km:
                earthquakes.append({
                    "magnitude": props.get("mag"),
                    "place": props.get("place"),
                    "time_utc": datetime.utcfromtimestamp(props["time"] / 1000).isoformat() + "Z",
                    "distance_km": round(distance, 2),
                    "url": props.get("url")
                })

        return {
            "location": {"latitude": latitude, "longitude": longitude, "radius_km": radius_km},
            "earthquakes": earthquakes,
            "count": len(earthquakes)
        }

    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch earthquake data: {e}")
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Error processing earthquake data: {e}")
    
@app.get("/hazard_check")

async def hazard_check(db: Session = Depends(get_db)):
    unsafe_users = await get_unsafe_users_with_coords(db)

    if not unsafe_users:
        return {"message": "No unsafe users currently."}

    hazards = []

    for u in unsafe_users:
        lat = u["lat"]
        lon = u["lon"]
        user = u["user"]

        weather = await get_weather_data(lat, lon)
        earthquake = await get_earthquake_data(lat, lon)
   
    hazards.append({
        "user_id": user.id,
        "user_name": getattr(user, "first_name", "") + " " + getattr(user, "last_name", ""),
        "location": {"lat": lat, "lon": lon},
        "hazards": {
            "weather": weather,
            "earthquakes": earthquake["earthquakes"]
                }
            })
    return {   "hazard_checks": hazards}


@app.post(
    "/ai/reason",
    operation_id="ai_reason",
    description="AI analyzes and provides reasoning for user safety status.",
    tags=["ai"],
)
async def ai_reason(request: AIReasonRequest):
    # 1. Fetch user data
    user = await get_unsafe_users(request.userId)
    lat = user["latitude_loc"]
    lon = user["longitude_loc"]

    # 2. Fetch hazard data using updated async functions
    weather = await get_weather_data(lat, lon)
    earthquake = await get_earthquake_data(lat, lon)

    # 3. Build the AI prompt
    prompt = f"""
You are a hazard-analysis assistant. Analyze the following user data and hazard data, 
then respond with a single JSON object ONLY with keys: risk_level, summary, recommended_action.

USER DATA:
{json.dumps(user)}

WEATHER DATA:
{json.dumps(weather)}

EARTHQUAKE DATA:
{json.dumps(earthquake)}
"""

    # 4. Prepare Ollama payload
    ollama_payload = {
        "model": "llama3.1",   # change if needed
        "messages": [
            {"role": "system", "content": "You are a precise hazard assessment AI. Reply ONLY with JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 800
    }

    # 5. Call Ollama local API
    try:
        res = requests.post("http://localhost:11434/api/chat", json=ollama_payload, timeout=60)
        res.raise_for_status()
        data = res.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Ollama call failed: {e}")
    except ValueError:
        raise HTTPException(status_code=500, detail="Ollama returned invalid JSON.")

    # 6. Extract assistant's JSON from response
    ai_text = None
    if isinstance(data, dict):
        if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
            ai_text = data["message"]["content"]
        elif "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if isinstance(choice.get("message"), dict) and "content" in choice["message"]:
                ai_text = choice["message"]["content"]
            elif "text" in choice:
                ai_text = choice["text"]

    if ai_text is None:
        ai_text = json.dumps(data)

    # 7. Clean and parse JSON
    cleaned = re.sub(r"```(?:json)?\n", "", ai_text, flags=re.IGNORECASE)
    cleaned = re.sub(r"```", "", cleaned)
    match = re.search(r"\{(?:[^{}]|(?R))*\}", cleaned)
    json_str = match.group(0) if match else cleaned.strip()

    try:
        ai_json = json.loads(json_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI JSON: {e}. Raw text: {cleaned[:1000]}")

    # 8. Validate required keys
    required = {"risk_level", "summary", "recommended_action"}
    if not required.issubset(ai_json.keys()):
        raise HTTPException(status_code=500, detail=f"AI JSON missing required keys: {list(ai_json.keys())}")

    # 9. Return structured output
    return {
        "user": user,
        "hazard_raw": {
            "weather": weather,
            "earthquake": earthquake
        },
        "analysis": ai_json
    }


@app.post("/ai/reason_and_rescue")
async def ai_reason_and_rescue(request: AIReasonRequest, db: Session = Depends(get_db)):
    # 1. Fetch user
    user = await get_unsafe_users(request.userId)
    lat, lon = user["latitude_loc"], user["longitude_loc"]

    # 2. Hazard check
    weather = await get_weather_data(lat, lon)
    earthquake = await get_earthquake_data(lat, lon)

    # 3. AI reasoning
    ai_response = await ai_reason(request)  # reuse your existing function
    risk_level = ai_response["analysis"]["risk_level"].lower()

    # 4. Only create rescue if HIGH risk
    if risk_level != "high":
        return {"user": user, "hazard_raw": {"weather": weather, "earthquake": earthquake}, "analysis": ai_response["analysis"], "rescue_created": False}

    # 5. Find nearest available rescuer
    nearest_rescuer = None
    nearest_dist = float("inf")

    available_rescuers = db.query(RescuerStatus).filter(RescuerStatus.is_available == True).all()
    if not available_rescuers:
        return {
            "user": user,
            "hazard_raw": {"weather": weather, "earthquake": earthquake},
            "analysis": ai_response["analysis"],
            "rescue_created": False,
            "message": "No available rescuer nearby."
        }


    for rescuer_status in available_rescuers:
        # Get rescuer user info
        rescuer_user = db.query(User).filter(User.id == rescuer_status.rescuer_id).first()
        if not rescuer_user:
            continue

        # Compute distance
        if rescuer_status.last_known_location:
            # Assuming POINT(lon lat) format in string
            match = re.search(r"POINT\(([-\d\.]+) ([-\d\.]+)\)", rescuer_status.last_known_location)
            if match:
                r_lon, r_lat = float(match.group(1)), float(match.group(2))
                dist = geodesic((lat, lon), (r_lat, r_lon)).km
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_rescuer = rescuer_status

    if not nearest_rescuer:
        return {
            "user": user,
            "hazard_raw": {"weather": weather, "earthquake": earthquake},
            "analysis": ai_response["analysis"],
            "rescue_created": False,
            "message": "No rescuer with valid location."
        }

    # 6. Create a rescue task (save in database if you have a table, here just a dict)
    new_task = {
        "taskId": f"task-{datetime.utcnow().timestamp()}",
        "unsafe_user": str(user.id),
        "rescuer_assigned": str(nearest_rescuer.rescuer_id),
        "hazard_level": risk_level,
        "location": {"lat": lat, "lon": lon},
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    # 9. Return full info including AI reasoning and rescue task
    return {
        "user": user,
        "hazard_raw": {"weather": weather, "earthquake": earthquake},
        "analysis": ai_response["analysis"],
        "rescue_created": True,
        "task": new_task,
        "rescuer_assigned": nearest_rescuer.rescuer_id
    }

# --- MCP Integration ---
mcp = FastApiMCP(
    app,
    name="Apxml API Services",
    description="Tools for disaster risk response.",
    describe_all_responses=True,    
    describe_full_response_schema=True,
    # Only expose the endpoints with these operation_ids 
    include_operations=[
        "get_user_details",
        "get_weather_data",
        "get_earthquake_data",
        "ai_reason",
        "ai_reason_and_rescue",


    ],
)

# Mount the MCP server on a specific path
mcp.mount(mount_path="/disaster_mcp")

# This final call scans the app for routes and finalizes the MCP setup.
# It must be called *after* all routes and the MCP mount are defined.
mcp.setup_server()
