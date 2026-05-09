# v4 Price + Fee Savings Analysis

## Per-incident table (v4 = Central case, the most realistic estimate)

| # | Incident | Side | Spread | Actual fill | v4 Central est | Delta price/ETH | Delta fees | Total $ saved |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-04-30 12:01 UTC | BUY | 14.0bps | $2261.43 | $2261.07 | $+0.36 | $+1.08 | $+2.97 |
| 2 | 2026-04-30 23:01-02 UTC | BUY | 24.4bps | $2258.00 | $2257.25 | $+0.75 | $+1.08 | $+5.06 |
| 3 | 2026-05-05 15:00 UTC | BUY | 15.0bps | $2377.05 | $2380.57 | $-3.52 | $+1.26 | $-19.48 |
| 4 | 2026-05-05 23:36 UTC | BUY | 6.7bps | $2356.08 | $2357.49 | $-1.41 | $+1.26 | $-7.13 |
| 5 | 2026-05-06 21:00 UTC | SELL | 3.8bps | $2350.49 | _v4 doesn't apply_ | — | — | $0 |
| 6 | 2026-05-08 23:02-05 UTC (May 9 01:02 LOCAL) | BUY | 13.0bps | $2311.34 | $2312.14 | $-0.80 | $+3.53 | $-1.30 |

**Total dollar savings across BUY incidents (Central case): $-19.88**

## Per-incident: Full scenario sweep (Optimistic / Central / Pessimistic)

| Incident | Side | Spread | Actual fill | v4 Opt | v4 Cen | v4 Pes |
|---|---|---|---|---|---|---|
| 2026-04-30 12:01 UTC | BUY | 14.0bps | $2261.43 | $2258.74 | $2261.07 | $2262.50 |
| 2026-04-30 23:01-02 UTC | BUY | 24.4bps | $2258.00 | $2253.87 | $2257.25 | $2259.32 |
| 2026-05-05 15:00 UTC | BUY | 15.0bps | $2377.05 | $2378.01 | $2380.57 | $2382.14 |
| 2026-05-05 23:36 UTC | BUY | 6.7bps | $2356.08 | $2355.83 | $2357.49 | $2358.51 |
| 2026-05-06 21:00 UTC | SELL | 3.8bps | $2350.49 | n/a | n/a | n/a |
| 2026-05-08 23:02-05 UTC (May 9 01:02 LOCAL) | BUY | 13.0bps | $2311.34 | $2309.86 | $2312.14 | $2313.53 |

## Methodology notes
- v4 maker fills at bid+0.01 (effectively the bid)
- v4 taker fills at ask + 0.09% Revolut taker fee
- Central case (55/45 maker/taker) is the most realistic per the v4 sim doc
- "Actual fees" estimate: 73% taker for the May 9 incident (from v4 doc), 55% for other
Traceback (most recent call last):
  File "C:\Users\Alex\algo_trading\engine\tools\v4_price_savings_analysis.py", line 179, in <module>
    main()
    ~~~~^^
  File "C:\Users\Alex\algo_trading\engine\tools\v4_price_savings_analysis.py", line 172, in main
    print('  wide-spread incidents (assumed similar slide \u2192 reject pattern). This is approximate;')
    ~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Alex\AppData\Local\Python\pythoncore-3.14-64\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2192' in position 47: character maps to <undefined>
