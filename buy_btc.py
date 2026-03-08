import base64, json, time, uuid, urllib.request, urllib.error, sys
from pathlib import Path
from nacl.signing import SigningKey
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

API_KEY = "REDACTED_API_KEY"
BASE_URL = "https://revx.revolut.com/api/1.0"

pk = serialization.load_pem_private_key(Path('private.pem').read_bytes(), password=None, backend=default_backend())
raw = pk.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
sk = SigningKey(raw)

def api(method, path, query='', body=None):
    body_str = json.dumps(body, separators=(',', ':')) if body else ''
    full_path = f"/api/1.0{path}"
    ts = str(int(time.time() * 1000))
    msg = f"{ts}{method}{full_path}{query}{body_str}".encode('utf-8')
    sig = base64.b64encode(sk.sign(msg).signature).decode()
    url = f"{BASE_URL}{path}"
    if query:
        url += f"?{query}"
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'X-Revx-Api-Key': API_KEY,
        'X-Revx-Timestamp': ts,
        'X-Revx-Signature': sig,
    }
    data = body_str.encode('utf-8') if body_str else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

print("=" * 60)
print("  BUY $100 BTC-USD (funded from CHF balance)")
print("=" * 60)

# Get current BTC price from orderbook
status, ob = api('GET', '/order-book/BTC-USD')
if status == 200:
    asks = ob.get('data', {}).get('asks', [])
    if asks:
        best_ask = asks[0].get('p', 'N/A')
        print(f"\n  BTC-USD best ask: ${best_ask}")

# Show balance
status, balances = api('GET', '/balances')
if status == 200:
    for b in balances:
        if float(b['total']) > 0:
            print(f"  Balance: {b['currency']} = {b['total']}")

# Confirm
print(f"\n  This will place a MARKET BUY for $100 of BTC-USD")
print(f"  Revolut X auto-converts CHF -> USD")
resp = input("\n  Type 'yes' to confirm: ").strip().lower()
if resp != 'yes':
    print("  Cancelled.")
    sys.exit(0)

# Place market order
order_body = {
    "client_order_id": str(uuid.uuid4()),
    "symbol": "BTC-USD",
    "side": "BUY",
    "order_configuration": {
        "market": {
            "quote_size": "100"
        }
    }
}
print(f"\n  Placing order: {json.dumps(order_body)}")

status, result = api('POST', '/orders', body=order_body)
print(f"\n  Status: {status}")
print(f"  Response: {json.dumps(result, indent=2)}")

if status == 200:
    print(f"\n  ORDER PLACED!")
    # Check updated balance
    time.sleep(2)
    status, balances = api('GET', '/balances')
    if status == 200:
        print(f"\n  Updated balances:")
        for b in balances:
            if float(b['total']) > 0:
                print(f"    {b['currency']:6s} = {b['total']}")
else:
    print(f"\n  Failed: {result.get('message', result)}")
