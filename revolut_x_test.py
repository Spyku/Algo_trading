"""
revolut_x_test.py - Revolut X API Connection Test (v5)
Base URL: https://revx.revolut.com/api/1.0
"""
import base64, json, time, urllib.request, urllib.error
from pathlib import Path

try:
    from nacl.signing import SigningKey
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.backends import default_backend
except ImportError:
    print("pip install pynacl cryptography"); exit(1)

API_KEY = "REDACTED_API_KEY"
PRIVATE_KEY_PATH = "private.pem"
BASE_URL = "https://revx.revolut.com/api/1.0"

def load_private_key():
    pem_data = Path(PRIVATE_KEY_PATH).read_bytes()
    pk = serialization.load_pem_private_key(pem_data, password=None, backend=default_backend())
    raw = pk.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    return SigningKey(raw)

def sign_request(sk, method, path, query='', body=''):
    ts = str(int(time.time() * 1000))
    msg = f"{ts}{method}{path}{query}{body}".encode('utf-8')
    signed = sk.sign(msg)
    sig = base64.b64encode(signed.signature).decode()
    return ts, sig

def do_request(method, url, headers=None):
    req = urllib.request.Request(url, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as e:
        return 0, str(e)

def main():
    print("=" * 60)
    print("  REVOLUT X API — TEST v5")
    print(f"  Base URL: {BASE_URL}")
    print("=" * 60)

    if not Path(PRIVATE_KEY_PATH).exists():
        print(f"\n  ✗ {PRIVATE_KEY_PATH} not found!"); return

    sk = load_private_key()
    print(f"  ✓ Key loaded\n")

    # --- PUBLIC (no auth) ---
    print("  --- PUBLIC ---")
    for label, path in [
        ("Tickers", "/tickers"),
        ("Pairs", "/configuration/pairs"),
        ("Currencies", "/configuration/currencies"),
        ("Orderbook BTC-USD", "/public/order-book/BTC-USD"),
        ("Last trades", "/public/last-trades"),
    ]:
        url = f"{BASE_URL}{path}"
        headers = {'Accept': 'application/json'}
        status, body = do_request('GET', url, headers)
        is_json = False
        try:
            data = json.loads(body)
            is_json = True
        except: pass
        tag = "✓" if is_json and status == 200 else "✗"
        print(f"  {tag} {label:25s} → {status}")
        if is_json:
            preview = json.dumps(data, indent=2)[:200]
            print(f"    {preview}")
        elif body:
            print(f"    {body[:150]}")

    # --- AUTHENTICATED ---
    print("\n  --- AUTHENTICATED ---")
    for label, path in [
        ("Balances", "/balances"),
        ("Active orders", "/orders/active"),
        ("Historical orders", "/orders/historical"),
    ]:
        full_path = f"/api/1.0{path}"
        url = f"{BASE_URL}{path}"
        ts, sig = sign_request(sk, 'GET', full_path)
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'X-Revx-Api-Key': API_KEY,
            'X-Revx-Timestamp': ts,
            'X-Revx-Signature': sig,
        }
        status, body = do_request('GET', url, headers)
        is_json = False
        try:
            data = json.loads(body)
            is_json = True
        except: pass
        tag = "✓" if is_json and status == 200 else "✗"
        print(f"  {tag} {label:25s} → {status}")
        if is_json:
            preview = json.dumps(data, indent=2)[:300]
            print(f"    {preview}")
        elif body:
            print(f"    {body[:150]}")

    print(f"\n{'='*60}")

if __name__ == '__main__':
    main()
