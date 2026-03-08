"""Quick debug: test all price fetch methods"""
import urllib.request, urllib.error, json

BASE = 'https://revx.revolut.com/api/1.0'

# Test 1: Public orderbook
print("=== Test 1: Public orderbook ===")
try:
    url = f'{BASE}/public/order-book/BTC-USD'
    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    resp = urllib.request.urlopen(req, timeout=10)
    data = json.loads(resp.read().decode())
    print(f"  Status: {resp.status}")
    print(f"  Data: {json.dumps(data)[:300]}")
except urllib.error.HTTPError as e:
    print(f"  HTTP Error: {e.code} {e.read().decode()[:200]}")
except Exception as e:
    print(f"  Error: {e}")

# Test 2: Authenticated tickers
print("\n=== Test 2: Authenticated tickers ===")
try:
    from crypto_revolut_trader import revx_api
    status, data = revx_api('GET', '/tickers')
    print(f"  Status: {status}")
    if status == 200:
        for t in data:
            if 'BTC' in t.get('symbol', ''):
                print(f"  BTC: {json.dumps(t)[:200]}")
                break
    else:
        print(f"  Data: {json.dumps(data)[:200]}")
except Exception as e:
    print(f"  Error: {e}")

# Test 3: Last trades
print("\n=== Test 3: Public last trades ===")
try:
    url = f'{BASE}/public/last-trades'
    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    resp = urllib.request.urlopen(req, timeout=10)
    data = json.loads(resp.read().decode())
    print(f"  Status: {resp.status}")
    print(f"  Data: {json.dumps(data)[:300]}")
except urllib.error.HTTPError as e:
    print(f"  HTTP Error: {e.code} {e.read().decode()[:200]}")
except Exception as e:
    print(f"  Error: {e}")

# Test 4: Authenticated last trades
print("\n=== Test 4: Auth last trades BTC-USD ===")
try:
    status, data = revx_api('GET', '/trades/private/BTC-USD')
    print(f"  Status: {status}")
    print(f"  Data: {json.dumps(data)[:300]}")
except Exception as e:
    print(f"  Error: {e}")
