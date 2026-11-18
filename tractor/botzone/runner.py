"""JSON first-turn envelope and subsequent bare-request keep-running loop."""

import json
import sys
from .inference import load
from .session import Session


def main():
    choose, config = load()
    session = Session(choose, config["capacity"])
    for line in sys.stdin:
        if not line.strip():
            continue
        message = json.loads(line)
        if "requests" in message:
            session = Session(choose, config["capacity"])
            requests, responses = message["requests"], message.get("responses", [])
            if len(requests) != len(responses) + 1:
                raise ValueError("Malformed history envelope")
            for request, response in zip(requests[:-1], responses):
                session.handle(request, forced=response)
            request = requests[-1]
        else:
            request = message
        response = session.handle(request)
        print(json.dumps({"response": response}, separators=(",", ":")))
        print(">>>BOTZONE_REQUEST_KEEP_RUNNING<<<")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
