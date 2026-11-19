"""Botzone JSON/simple IO, cold-history replay and keep-running mode.

Recommended submission: Python 3.6, JSON IO (SimpleIO unchecked).
"""
import json
import sys
from .inference import load
from .session import Session

KEEP_RUNNING = '>>>BOTZONE_REQUEST_KEEP_RUNNING<<<'


def _line(stream, what):
    value = stream.readline()
    if not value:
        raise ValueError('Incomplete Botzone input: missing ' + what)
    return value.rstrip('\r\n')


def _history(session, payload):
    requests, responses = payload.get('requests'), payload.get('responses', [])
    if not isinstance(requests, list) or not isinstance(responses, list):
        raise ValueError('JSON input requires requests/responses arrays')
    if len(requests) != len(responses) + 1:
        raise ValueError('Expected one current request after historical responses')
    if not all(isinstance(x, str) for x in requests + responses):
        raise ValueError('Mahjong requests and responses must be strings')
    for request, response in zip(requests[:-1], responses):
        session.handle(request, forced=response)
    return requests[-1]


def _initial(session, first, stream):
    if first.lstrip().startswith('{'):
        return 'json', _history(session, json.loads(first))
    try:
        turn = int(first)
    except ValueError:
        raise ValueError('Expected Botzone JSON envelope or SimpleIO turn number')
    if turn < 1:
        raise ValueError('SimpleIO turn number starts at 1')
    for _ in range(turn - 1):
        request = _line(stream, 'historical request')
        response = _line(stream, 'historical response')
        session.handle(request, forced=response)
    request = _line(stream, 'current request')
    # Official cold SimpleIO includes data and globaldata. Consume both before
    # request-only keep-running input, so nonempty metadata is never a move.
    # EOF is accepted for local one-shot fixtures. Live streams send both lines.
    stream.readline()
    stream.readline()
    return 'simple', request


def _next_request(line):
    line = line.strip()
    if line.startswith('"'):
        request = json.loads(line)
    elif line.startswith('{'):
        payload = json.loads(line)
        if 'request' in payload:
            request = payload['request']
        elif payload.get('responses', []) == [] and len(payload.get('requests', [])) == 1:
            request = payload['requests'][0]
        else:
            raise ValueError('Keep-running input must contain only the current request')
    else:
        request = line
    if not isinstance(request, str):
        raise ValueError('Mahjong request must be a string')
    return request


def main():
    first = sys.stdin.readline()
    if not first:
        return
    choose, config = load()
    session = Session(choose, feature_set=config.get('feature_set', 'basic'))
    mode, request = _initial(session, first, sys.stdin)
    keep_running = config.get('keep_running', True)
    debug = json.dumps(dict(model=config.get('model_id', config.get('feature_set')),
                            weight=config.get('weight_path'),
                            sha256=config.get('weight_sha256'),
                            shanten=config.get('shanten_backend', 'not_used')),
                       separators=(',', ':'))
    while True:
        response = session.handle(request)
        if mode == 'json':
            print(json.dumps(dict(response=response, debug=debug, data='', globaldata=''),
                             separators=(',', ':')))
        else:
            print(response)
            print(debug)
            print('')
            print('')
        if keep_running:
            print(KEEP_RUNNING)
        sys.stdout.flush()
        if not keep_running:
            return
        line = sys.stdin.readline()
        if not line:
            return
        request = _next_request(line)
        debug = ''


if __name__ == '__main__':
    main()
