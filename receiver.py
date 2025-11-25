import asyncio

from alay import AIReasonRequest, ai_reason_and_rescue, fetch_unsafe_users_api


async def agent_loop():
    while True:
        print("\n===== AGENTIC AI CYCLE START =====")

        # Step 1: Get unsafe users
        unsafe_users = await fetch_unsafe_users_api()
        if not unsafe_users:
            print("[AI] No unsafe users found.")
        else:
            print(f"[AI] Found {len(unsafe_users)} unsafe users")

        # Step 2: Process each user for hazard analysis and rescue
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

            except Exception as e:
                print(f"[AI] Error processing user {user_id}: {e}")

        print("===== AGENTIC AI CYCLE END – Waiting 60s =====\n")
        await asyncio.sleep(60)
