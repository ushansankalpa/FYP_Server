# This file is responsible for signing , encoding , decoding and returning JWTS
import time
from typing import Dict
import secrets

import jwt
from decouple import config

from schemas.user import UserSchema

JWT_SECRET = secrets.token_hex(10)
JWT_ALGORITHM = "HS256"


def token_response(token: str):
    return {
        "access_token": token
    }

# function used for signing the JWT string
def signJWT(user: dict) -> Dict[str, str]:
    payload = {
        "id": user['id'],
        "fullname": user['fullname'],
        "email": user['email'],
        "role": user['role'],
        "expires": time.time() + 600
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return token_response(token)


def decodeJWT(token: str) -> dict:
    try:
        decoded_token = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return decoded_token if decoded_token["expires"] >= time.time() else None
    except:
        return {}
