#!/usr/bin/env python3
"""A real local OPC-UA server for live integration testing of
``pinneapple_systems.digital_twin.io.stream.OPCUAStream`` -- not a mock
inside the test process, a genuine ``asyncua`` server (the same
production-grade library used by the real client) listening on a real
TCP port, with two tag values that actually change over time (a sine
wave "Temperature" and a linearly ramping "Pressure"), so a polling
client sees real, changing data -- not a frozen constant.

Usage:
    python3 tests/simulators/opcua_simulator.py [--port 4855]

Prints the two node ids it creates (needed by ``OPCUAStream``'s
``node_ids`` parameter) and then blocks, serving forever, until
interrupted (Ctrl-C) or killed.
"""
from __future__ import annotations

import argparse
import math
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=4855)
    ap.add_argument("--update-interval", type=float, default=0.2)
    args = ap.parse_args()

    from asyncua.sync import Server

    server = Server()
    server.set_endpoint(f"opc.tcp://127.0.0.1:{args.port}/pinneapple/simulator/")
    idx = server.register_namespace("http://pinneapple.simulator")

    sensor = server.nodes.objects.add_object(idx, "Sensor1")
    temperature = sensor.add_variable(idx, "Temperature", 20.0)
    pressure = sensor.add_variable(idx, "Pressure", 101.3)
    temperature.set_writable()
    pressure.set_writable()

    print(f"Temperature node id: {temperature.nodeid.to_string()}", flush=True)
    print(f"Pressure node id: {pressure.nodeid.to_string()}", flush=True)

    server.start()
    print(f"OPC-UA simulator listening on opc.tcp://127.0.0.1:{args.port}/pinneapple/simulator/", flush=True)

    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            temperature.write_value(20.0 + 3.0 * math.sin(t / 5.0))
            pressure.write_value(101.3 + 0.01 * t)
            time.sleep(args.update_interval)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
