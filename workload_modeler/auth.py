"""Signed browser sessions; tokens are never stored in browser storage or URLs."""
import hashlib
import hmac
import time


def session_cookie(token, expiry):
    value = str(expiry)
    return value + '.' + hmac.new(token.encode(), value.encode(), hashlib.sha256).hexdigest()


def valid_cookie(token, value):
    if not token or not value:
        return False
    try:
        expiry = int(value.split('.', 1)[0])
        return expiry > time.time() and hmac.compare_digest(session_cookie(token, expiry).encode(), value.encode())
    except (ValueError, TypeError):
        return False
