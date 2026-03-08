import base64, json, time, urllib.request, urllib.error
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
print("  TRADE DETAILS & FEES")
print("=" * 60)

# Check historical orders
print("\n  --- Recent Orders ---")
status, orders = api('GET', '/orders/historical')
if status == 200:
    for o in orders.get('data', []):
        print(f"\n  Order: {o.get('venue_order_id', 'N/A')}")
        print(f"    Symbol: {o.get('symbol', 'N/A')}")
        print(f"    Side: {o.get('side', 'N/A')}")
        print(f"    State: {o.get('state', 'N/A')}")
        print(f"    Type: {o.get('order_type', 'N/A')}")
        print(f"    Created: {o.get('created_at', 'N/A')}")
        
        # Print all fields to find fee info
        for k, v in o.items():
            if k not in ('venue_order_id', 'symbol', 'side', 'state', 'order_type', 'created_at'):
                print(f"    {k}: {v}")

        # Get fills for this order
        oid = o.get('venue_order_id')
        if oid:
            print(f"\n    --- Fills ---")
            fstatus, fills = api('GET', f'/orders/fills/{oid}')
            if fstatus == 200:
                print(f"    {json.dumps(fills, indent=4)}")
            else:
                print(f"    Fills error: {fstatus} {fills}")
else:
    print(f"  Error: {status} {orders}")

# Check private trades
print("\n\n  --- Private Trades BTC-USD ---")
status, trades = api('GET', '/trades/private/BTC-USD')
if status == 200:
    print(f"  {json.dumps(trades, indent=2)}")
else:
    print(f"  Error: {status} {trades}")

# Current balances
print("\n\n  --- Current Balances ---")
status, balances = api('GET', '/balances')
if status == 200:
    for b in balances:
        if float(b['total']) > 0:
            print(f"  {b['currency']:6s}  available={b['available']:>15s}  total={b['total']:>15s}")
