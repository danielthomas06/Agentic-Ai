import requests


API_URL = "http://127.0.0.1:8000/agent/run"


def main():
    payload = {
        "goal": "Explain visual odometry for GPS-denied drone navigation"
    }

    response = requests.post(
        API_URL,
        json=payload,
        timeout=600,
    )

    response.raise_for_status()

    result = response.json()

    print("\nAGENT RESPONSE")
    print("=" * 60)
    print(result["answer"])

    print("\nMETADATA")
    print("=" * 60)
    print(f"Agents used: {result['agents_used']}")
    print(f"Iterations:  {result['iterations']}")
    print(f"Success:     {result['success']}")


if __name__ == "__main__":
    main()