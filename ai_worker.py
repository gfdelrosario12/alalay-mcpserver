import time
import requests
import json

BACKEND = "http://localhost:8080/disaster_mcp"   # your FastAPI URL

def check_user(user_id: str):
    """Call your backend AI hazard assessment."""
    try:
        response = requests.post(
            f"{BACKEND}/ai/reason",
            json={"userId": user_id},
            timeout=60
        )
        response.raise_for_status()
        return response.json()

    except Exception as e:
        print(f"[ERROR] Failed to process {user_id}: {e}")
        return None


def main_loop():
    print("🤖 AI Worker Running...")

    # Example: these would come from your DB
    monitored_users = ["user123", "user456", "user789"]

    while True:
        print("\n⏳ Checking hazards...")

        for user in monitored_users:
            print(f"➡ Checking {user}...")

            result = check_user(user)

            if result:
                print(f"📌 RESULT FOR {user}:")
                print(json.dumps(result["analysis"], indent=2))

                # TODO: Optionally call backend to save AI decision
                # requests.post(f"{BACKEND}/save_decision", json=result)

        print("🕒 Done. Sleeping 60 seconds...\n")
        time.sleep(60)


if __name__ == "__main__":
    main_loop()
