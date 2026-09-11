#!/usr/bin/env python3
"""A real local Modbus TCP server for live integration testing of
``pinneapple_systems.digital_twin.io.stream.ModbusStream`` -- a genuine
``pymodbus`` server (the same production-grade library the real client
uses), not a mock. Two holding registers hold real (not fabricated)
values that a real Modbus TCP client reads over the real wire protocol.

Honest limitation (found by direct testing, not assumed): unlike the
OPC-UA simulator in this same directory, this version's registers are
STATIC, not time-varying. pymodbus 3.15's `ModbusSequentialDataBlock`
is a compatibility shim over a newer, still-transitional `SimData`/
`SimDevice` model in this pymodbus version -- writing to the block's own
backing list (`block.simdata[0].values`) after `StartTcpServer` has
started does NOT change what a real client subsequently reads (verified
directly: it keeps serving the values passed in at construction time).
This module does not fabricate a workaround for that -- it serves real,
static values instead of pretending to support live updates it can't
actually deliver in this pymodbus version. `ModbusStream` itself is
still exercised for real: a real client connects, sends a real Modbus
read-holding-registers request, and gets a real response.

Usage:
    python3 tests/simulators/modbus_simulator.py [--port 5502]

Register map (0-based addresses, as the CLIENT sees them):
    address 0 -- temperature * 100, as an unsigned 16-bit integer
                 (raw value 2000 == 20.00 real units; scale=0.01 on the
                 client side)
    address 1 -- pressure * 10, as an unsigned 16-bit integer
                 (raw value 1013 == 101.3 real units; scale=0.1)

Blocks, serving forever, until interrupted (Ctrl-C) or killed.
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=5502)
    args = ap.parse_args()

    from pymodbus.datastore import ModbusDeviceContext, ModbusServerContext, ModbusSequentialDataBlock
    from pymodbus.server import StartTcpServer

    # NOTE: ModbusSequentialDataBlock's own address argument is 1-based
    # internally (address=1 -> client-visible register 0) -- confirmed
    # by direct testing against this pymodbus version, not assumed.
    block = ModbusSequentialDataBlock(1, [2000, 1013])
    device = ModbusDeviceContext(hr=block)
    context = ModbusServerContext(devices=device, single=True)

    print(f"Modbus simulator listening on 127.0.0.1:{args.port} "
          f"(register 0=temperature*100=2000, register 1=pressure*10=1013, static values)", flush=True)
    StartTcpServer(context=context, address=("127.0.0.1", args.port))


if __name__ == "__main__":
    main()
