import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, Depends, HTTPException, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi_mcp import FastApiMCP, AuthConfig
import random

# Load environment variables from the .env file
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Apxml Awesome MCP",
    description="API services for internal use.",
)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Server is running"}

# --- Tool endpoints ---

class Order(BaseModel):
    order_id: str
    customer_id: int
    status: str
    total_amount: float
    currency: str

class Customer(BaseModel):
    customer_id: int
    name: str
    email: str
    join_date: str

@app.get(
    "/orders/{order_id}",
    operation_id="get_order_details",
    description="Fetches detailed information for a specific order by its ID.",
    tags=["orders"],
    response_model=Order,
)
async def get_order_details(order_id: str) -> Dict[str, Any]:
    """Simulate fetching order data."""
    if order_id == "ORD12345":
        return {
            "order_id": order_id,
            "customer_id": 101,
            "status": "shipped",
            "total_amount": 199.99,
            "currency": "USD",
        }
    raise HTTPException(status_code=404, detail="Order not found")

@app.post(
    "/orders/random",
    operation_id="make_order_random",
    description="Generates a random order for testing purposes.",
    tags=["orders"],
    response_model=Order,
)
async def make_order_random() -> Order:
    order_id = f"ORD{random.randint(10000, 99999)}"
    return Order(
        order_id=order_id,
        customer_id=random.randint(100, 200),
        status="processing",
        total_amount=round(random.uniform(20.0, 500.0), 2),
        currency="USD",
    )

@app.get(
    "/customers/{customer_id}",
    operation_id="get_customer_profile",
    description="Retrieves the profile of a customer by their unique ID.",
    tags=["customers"],
    response_model=Customer,
)
async def get_customer_profile(customer_id: int) -> Dict[str, Any]:
    """Simulate fetching customer data."""
    if customer_id == 101:
        return {
            "customer_id": customer_id,
            "name": "Jane Doe",
            "email": "jane.doe@example.com",
            "join_date": "2023-05-20",
        }
    raise HTTPException(status_code=404, detail="Customer not found")

# --- Authentication dependency ---

def verify_auth(request: Request):
    """Dependency to verify the internal bearer token."""
    auth_header: Optional[str] = request.headers.get("Authorization")
    if auth_header is None or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = auth_header.split(" ")[1]
    expected_token = os.getenv("INTERNAL_TOKEN")
    if token != expected_token:
        raise HTTPException(status_code=403, detail="Forbidden")

# --- MCP Integration ---

mcp = FastApiMCP(
    app,
    name="Apxml API Services",
    description="Tools for managing orders and customers.",
    describe_all_responses=True,
    describe_full_response_schema=True,
    include_operations=[
        "get_order_details",
        "get_customer_profile",
        "make_order_random",
    ],
)

# Mount MCP on /mcp
mcp.mount(mount_path="/mcp")

# Finalize MCP setup
mcp.setup_server()
