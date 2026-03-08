import base64, json, time, urllib.request
from pathlib import Path
from nacl.signing import SigningKey
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

pk = serialization.load_pem_private_key(Path('private.pem').read_bytes(), password=None, backend=default_backend())
raw = pk.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
sk = SigningKey(raw)

path = '/api/1.0/balances'
ts = str(int(time.time() * 1000))
sig = base64.b64encode(sk.sign(f'{ts}GET{path}'.encode()).signature).decode()

req = urllib.request.Request('https://revx.revolut.com/api/1.0/balances', headers={
    'Accept': 'application/json',
    'X-Revx-Api-Key': 'REDACTED_API_KEY',
    'X-Revx-Timestamp': ts,
    'X-Revx-Signature': sig,
})
data = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())

print("\nALL BALANCES:")
for b in data:
    marker = " <<<" if float(b['total']) > 0 else ""
    print(f"  {b['currency']:6s}  available={b['available']:>15s}  total={b['total']:>15s}{marker}")

non_zero = [b for b in data if float(b['total']) > 0]
if non_zero:
    print(f"\nFunded currencies: {', '.join(b['currency'] for b in non_zero)}")
else:
    print("\nNo funds found in any currency.")
