"""Backtest V5 hourly + V15 15-min combination strategies for BTC."""
import json

FEE = 0.0011

# Load both signal sources
with open('models/crypto_hourly_chart_data.json') as f:
    v5_data = json.load(f)
with open('models/crypto_15m_chart_data.json') as f:
    v15_data = json.load(f)

v5_sigs = v5_data['assets']['BTC']
v15_sigs = v15_data['assets']['BTC']

v5_map = {s['datetime']: s for s in v5_sigs}
v15_map = {s['datetime']: s for s in v15_sigs}

v5_times = sorted(v5_map.keys())
v15_times = sorted(v15_map.keys())
overlap_start = max(v5_times[0], v15_times[0])
overlap_end = min(v5_times[-1], v15_times[-1])

print(f"V5 range:  {v5_times[0]} to {v5_times[-1]} ({len(v5_times)} hourly)")
print(f"V15 range: {v15_times[0]} to {v15_times[-1]} ({len(v15_times)} x 15min)")
print(f"Overlap:   {overlap_start} to {overlap_end}")


def simulate(timeline, name):
    cash, held, in_pos, entry_px = 1000.0, 0.0, False, 0.0
    trades, wins, losses = 0, 0, 0
    peak = 1000.0
    max_dd = 0.0

    for t in timeline:
        signal = t['combined_signal']
        price = t['price']

        if signal == 'BUY' and not in_pos:
            held = cash * (1 - FEE) / price
            cash = 0
            in_pos = True
            entry_px = price
            trades += 1
        elif signal == 'SELL' and in_pos:
            cash = held * price * (1 - FEE)
            if price > entry_px:
                wins += 1
            else:
                losses += 1
            held = 0
            in_pos = False

        pv = cash if not in_pos else held * price
        if pv > peak:
            peak = pv
        dd = (peak - pv) / peak * 100
        if dd > max_dd:
            max_dd = dd

    if in_pos:
        last_price = timeline[-1]['price']
        cash = held * last_price * (1 - FEE)
        if last_price > entry_px:
            wins += 1
        else:
            losses += 1

    ret = (cash / 1000.0 - 1) * 100
    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0
    print(f"  {name:<35s} ret={ret:>+7.1f}%  wr={wr:>5.1f}%  trades={trades:>4d}  maxDD={max_dd:>5.1f}%")
    return ret


# Build unified 15-min timeline over overlap
overlap_v15 = [s for s in v15_sigs if overlap_start <= s['datetime'] <= overlap_end]

unified = []
for s15 in overlap_v15:
    dt = s15['datetime']
    hour_key = dt[:13] + ":00"
    v5 = v5_map.get(hour_key)

    unified.append({
        'datetime': dt,
        'price': s15['close'],
        'v15_signal': s15['signal'],
        'v15_conf': s15['confidence'],
        'v5_signal': v5['signal'] if v5 else 'HOLD',
        'v5_conf': v5['confidence'] if v5 else 50,
    })

print(f"\nUnified timeline: {len(unified)} slots ({unified[0]['datetime']} to {unified[-1]['datetime']})")
bh = (unified[-1]['price'] / unified[0]['price'] - 1) * 100
print(f"Buy & Hold: {bh:+.1f}%\n")

# --- Strategies ---

# 1. V5 4h_only @80%
for u in unified:
    if u['v5_signal'] == 'SELL':
        u['combined_signal'] = 'SELL'
    elif u['v5_signal'] == 'BUY' and u['v5_conf'] >= 80 and u['datetime'].endswith(':00'):
        u['combined_signal'] = 'BUY'
    else:
        u['combined_signal'] = 'HOLD'
simulate(unified, "1. V5 4h_only @80%")

# 1b. V5 4h_only @90%
for u in unified:
    if u['v5_signal'] == 'SELL':
        u['combined_signal'] = 'SELL'
    elif u['v5_signal'] == 'BUY' and u['v5_conf'] >= 90 and u['datetime'].endswith(':00'):
        u['combined_signal'] = 'BUY'
    else:
        u['combined_signal'] = 'HOLD'
simulate(unified, "1b. V5 4h_only @90%")

# 2. V15 alone (raw, no conf filter)
for u in unified:
    u['combined_signal'] = u['v15_signal']
simulate(unified, "2. V15 alone (raw)")

# 2b. V15 @61% (Mode F optimal)
for u in unified:
    if u['v15_signal'] == 'SELL':
        u['combined_signal'] = 'SELL'
    elif u['v15_signal'] == 'BUY' and u['v15_conf'] >= 61:
        u['combined_signal'] = 'BUY'
    else:
        u['combined_signal'] = 'HOLD'
simulate(unified, "2b. V15 @61% (Mode F)")

# 3. V15 override: follow V5, but V15 overrides if conf>=80%
for u in unified:
    if u['v5_signal'] == 'SELL':
        base = 'SELL'
    elif u['v5_signal'] == 'BUY' and u['v5_conf'] >= 80:
        base = 'BUY'
    else:
        base = 'HOLD'
    if u['v15_conf'] >= 80:
        if u['v15_signal'] == 'SELL' and base != 'SELL':
            base = 'SELL'
        elif u['v15_signal'] == 'BUY' and base != 'BUY':
            base = 'BUY'
    u['combined_signal'] = base
simulate(unified, "3. V5 + V15 override @80%")

# 4. V15 confirmation: V5 BUY only if V15 agrees
for u in unified:
    if u['v5_signal'] == 'SELL' or u['v15_signal'] == 'SELL':
        u['combined_signal'] = 'SELL'
    elif u['v5_signal'] == 'BUY' and u['v5_conf'] >= 80 and u['v15_signal'] == 'BUY' and u['datetime'].endswith(':00'):
        u['combined_signal'] = 'BUY'
    else:
        u['combined_signal'] = 'HOLD'
simulate(unified, "4. V5 BUY needs V15 confirm")

# 5a. V15 entries, V5 exits
for u in unified:
    if u['v5_signal'] == 'SELL':
        u['combined_signal'] = 'SELL'
    elif u['v15_signal'] == 'BUY' and u['v15_conf'] >= 61:
        u['combined_signal'] = 'BUY'
    else:
        u['combined_signal'] = 'HOLD'
simulate(unified, "5a. V15 entry, V5 exit")

# 5b. V5 entries, V15 exits
for u in unified:
    if u['v15_signal'] == 'SELL':
        u['combined_signal'] = 'SELL'
    elif u['v5_signal'] == 'BUY' and u['v5_conf'] >= 80 and u['datetime'].endswith(':00'):
        u['combined_signal'] = 'BUY'
    else:
        u['combined_signal'] = 'HOLD'
simulate(unified, "5b. V5 entry, V15 exit")

# 6. Either BUY = enter, either SELL = exit
for u in unified:
    if u['v5_signal'] == 'SELL' or u['v15_signal'] == 'SELL':
        u['combined_signal'] = 'SELL'
    elif (u['v5_signal'] == 'BUY' and u['v5_conf'] >= 80) or (u['v15_signal'] == 'BUY' and u['v15_conf'] >= 61):
        u['combined_signal'] = 'BUY'
    else:
        u['combined_signal'] = 'HOLD'
simulate(unified, "6. Either BUY, either SELL")

# 7. Both agree = enter, either SELL = exit
for u in unified:
    if u['v5_signal'] == 'SELL' or u['v15_signal'] == 'SELL':
        u['combined_signal'] = 'SELL'
    elif u['v5_signal'] == 'BUY' and u['v5_conf'] >= 80 and u['v15_signal'] == 'BUY' and u['v15_conf'] >= 61 and u['datetime'].endswith(':00'):
        u['combined_signal'] = 'BUY'
    else:
        u['combined_signal'] = 'HOLD'
simulate(unified, "7. Both agree BUY, either SELL")
