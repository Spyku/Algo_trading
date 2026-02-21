"""
IB Connection Test
===================
Run this first to verify your TWS/IB Gateway setup works.
Tests: connection, account info, contract resolution, market data.

Usage:
  python ib_test_connection.py
"""

import sys
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings('ignore')

def test_ib_connection():
    print("=" * 60)
    print("  IB CONNECTION TEST")
    print("=" * 60)

    # Step 1: Check ib_insync is installed
    print("\n1. Checking ib_insync installation...")
    try:
        # Python 3.14 fix: always create event loop before importing ib_insync
        import asyncio
        asyncio.set_event_loop(asyncio.new_event_loop())

        from ib_insync import IB, CFD, MarketOrder, StopOrder
        print("   OK - ib_insync installed")
    except ImportError:
        print("   FAILED - ib_insync not installed!")
        print("   Run: pip install ib_insync")
        return False

    # Step 2: Connect to IB
    print("\n2. Connecting to IB (127.0.0.1:7497, paper trading)...")
    ib = IB()
    try:
        ib.connect('127.0.0.1', 4002, clientId=99)
        print("   OK - Connected!")
    except Exception as e:
        print(f"   FAILED - {e}")
        print("\n   Checklist:")
        print("   - Is TWS or IB Gateway running?")
        print("   - File > Global Configuration > API > Settings:")
        print("     * 'Enable ActiveX and Socket Clients' = checked")
        print("     * 'Socket port' = 7497 (paper)")
        print("     * 'Allow connections from localhost only' = checked")
        print("     * 'Read-Only API' = UNCHECKED (need write for orders)")
        print("   - Is another client already using clientId 99?")
        return False

    # Step 3: Account info
    print("\n3. Account Information...")
    try:
        summary = ib.accountSummary()
        for av in summary:
            if av.tag in ('NetLiquidation', 'TotalCashValue', 'BuyingPower',
                          'AvailableFunds', 'Currency'):
                print(f"   {av.tag}: {av.currency} {av.value}")
        print("   OK - Account data received")
    except Exception as e:
        print(f"   WARNING - {e}")

    # Step 4: Test CFD contracts
    print("\n4. Testing CFD Contracts...")
    test_cfds = {
        'SMI':   ('IBCH20', 'SMART', 'CHF'),
        'DAX':   ('IBDE40', 'SMART', 'EUR'),
        'CAC40': ('IBFR40', 'SMART', 'EUR'),
    }

    for name, (symbol, exchange, currency) in test_cfds.items():
        try:
            contract = CFD(symbol=symbol, exchange=exchange, currency=currency)
            qualified = ib.qualifyContracts(contract)
            if qualified:
                c = qualified[0]
                print(f"   {name:6s} ({symbol}): OK - conId={c.conId}")
            else:
                print(f"   {name:6s} ({symbol}): WARNING - could not qualify")
                print(f"          You may need CFD trading permissions in your account")
        except Exception as e:
            print(f"   {name:6s} ({symbol}): FAILED - {e}")

    # Step 5: Test market data
    print("\n5. Testing Market Data...")
    for name, (symbol, exchange, currency) in test_cfds.items():
        try:
            contract = CFD(symbol=symbol, exchange=exchange, currency=currency)
            ib.qualifyContracts(contract)
            ticker = ib.reqMktData(contract, '', False, False)
            ib.sleep(3)

            price = ticker.marketPrice()
            bid = ticker.bid
            ask = ticker.ask
            last = ticker.last

            ib.cancelMktData(contract)

            if price and price == price:  # not NaN
                print(f"   {name:6s}: price={price:,.2f} "
                      f"bid={bid if bid == bid else 'N/A'} "
                      f"ask={ask if ask == ask else 'N/A'}")
            else:
                print(f"   {name:6s}: No live price (market may be closed)")
                if last == last:  # not NaN
                    print(f"          Last known: {last:,.2f}")
        except Exception as e:
            print(f"   {name:6s}: FAILED - {e}")

    # Step 6: Check existing positions
    print("\n6. Current Positions...")
    positions = ib.positions()
    if positions:
        for pos in positions:
            print(f"   {pos.contract.symbol}: {pos.position} units "
                  f"@ avg {pos.avgCost:,.2f}")
    else:
        print("   No open positions")

    # Step 7: Check open orders
    print("\n7. Open Orders...")
    orders = ib.openOrders()
    if orders:
        for order in orders:
            print(f"   {order.action} {order.totalQuantity} "
                  f"({order.orderType})")
    else:
        print("   No open orders")

    # Disconnect
    ib.disconnect()

    print("\n" + "=" * 60)
    print("  TEST COMPLETE")
    print("=" * 60)
    print("\n  Next steps:")
    print("  1. Make sure hourly_best_models.csv exists")
    print("     (run: python hourly_trading_system.py -> Mode A)")
    print("  2. Single test run:")
    print("     python ib_auto_trader.py")
    print("  3. Check status:")
    print("     python ib_auto_trader.py --status")
    print("  4. Continuous trading:")
    print("     python ib_auto_trader.py --loop")
    print("  5. Emergency close all:")
    print("     python ib_auto_trader.py --close-all")

    return True


if __name__ == '__main__':
    test_ib_connection()
