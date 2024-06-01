import time
import hmac
import base64
import hashlib
import json
import requests
import re
import maps
from api_details import kucoin_passport


class KucoinApi:
    def __init__(self):
        self.endpoints = maps.kucoin_endpoints

    def get_headers(self, request_type, endpoint, params=None):
        api_key = kucoin_passport['KEY']
        api_secret = kucoin_passport["SECRET"]
        api_passphrase = kucoin_passport['PASSPHRASE']
        now = int(time.time() * 1000)
        if params: 
            body = json.dumps(params)
        else:
            body = ""
        str_to_sign = str(now) + request_type + endpoint + body
        signature = base64.b64encode(hmac.new(
                api_secret.encode('utf-8'), 
                str_to_sign.encode('utf-8'), 
                hashlib.sha256).digest()
            )
        passphrase = base64.b64encode(hmac.new(
            api_secret.encode('utf-8'), 
            api_passphrase.encode('utf-8'), 
            hashlib.sha256).digest()
            )
        headers = {
            "KC-API-SIGN": signature,
            "KC-API-TIMESTAMP": str(now),
            "KC-API-KEY": api_key,
            "KC-API-PASSPHRASE": passphrase,
            "KC-API-KEY-VERSION": "2"
            }
        return headers
    
    def process_request(self, request_type, endpoint, **kwargs):
        if not kwargs or not list(kwargs.values())[0]:
            headers = self.get_headers(request_type, endpoint)
            return dict(requests.request(
                request_type, 
                self.endpoints['base'] + endpoint,
                headers=headers
                ).json())
        if endpoint[-1] == "/":
            params = {key: value for (key, value) in kwargs.items() if value}
            endpoint = endpoint + list(params.values())[0]
            headers = self.get_headers(request_type, endpoint, params)
            return dict(requests.request(
                request_type,
                self.endpoints['base'] + endpoint,
                json=params, 
                headers=headers
                ).json())
        remove_specials = lambda x: re.sub('\W+','', x)
        if request_type == "GET" or request_type == "DELETE":
            params = {f"&{key}=": value for (key, value) in kwargs.items() if value}
            keys = list(params.keys())
            values = list(params.values())
            keys[0] = f"?{keys[0][1:]}"
            params = dict(zip(keys, values))
            endpoint = endpoint + "".join("{}{}".format(*p) for p in params.items())
            params = {remove_specials(key): remove_specials(value) for (key, value) in params.items() if value}
            headers = self.get_headers(request_type, endpoint, params)
        else: 
            params = {key: value for (key, value) in kwargs.items() if value}
            headers = self.get_headers(request_type, endpoint, params)
        response = dict(requests.request(
            request_type, 
            self.endpoints['base'] + endpoint,
            json=params,
            headers=headers
            ).json())
        return response
    
