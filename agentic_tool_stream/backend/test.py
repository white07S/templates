import requests, json, sys, time

BASE_URL   = "http://localhost:8000"
USER_ID    = "user_42"
SESSION_ID = "session_791"                     # reuse to keep data consistent
QUERY      = "Do you know my name and what was my previous question?"

# ----------------------------------------------------------------------
# 1. /chat  – stream a single assistant response
# ----------------------------------------------------------------------
def test_chat():
    print("\n--- /chat (streaming) ---")
    payload = {"user_id": USER_ID,
               "session_id": SESSION_ID,
               "query": QUERY}

    with requests.post(f"{BASE_URL}/chat", json=payload, stream=True) as r:
        r.raise_for_status()
        # stream chunks immediately to stdout
        for chunk in r.iter_content(chunk_size=None):
            if chunk:
                sys.stdout.write(chunk.decode())
                sys.stdout.flush()
    print("\n--- end stream ---\n")

# ----------------------------------------------------------------------
# 2. /get-all-session-id  – list every session for USER_ID
# ----------------------------------------------------------------------
def test_get_all_session_id():
    print("\n--- /get-all-session-id ---")
    payload = {"user_id": USER_ID}
    r = requests.post(f"{BASE_URL}/get-all-session-id", json=payload)
    r.raise_for_status()
    data = r.json()
    print(json.dumps(data, indent=2))

    if SESSION_ID not in data.get("session_ids", []):
        print(f"⚠️  Expected {SESSION_ID} in session list!")

# ----------------------------------------------------------------------
# 3. /get-chat-details  – retrieve entire conversation for SESSION_ID
# ----------------------------------------------------------------------
def test_get_chat_details():
    print("\n--- /get-chat-details ---")
    payload = {"session_id": SESSION_ID}
    r = requests.post(f"{BASE_URL}/get-chat-details", json=payload)
    r.raise_for_status()
    data = r.json()
    print(json.dumps(data, indent=2))

# ----------------------------------------------------------------------
if __name__ == "__main__":
    test_chat()
    # Tiny, defensive pause to ensure TinyDB commit finishes
    time.sleep(0.5)
    test_get_all_session_id()
    test_get_chat_details()