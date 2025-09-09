# connection_test_simserv.py
from simserv_oven import SimServOven, SimServError

HOST = "192.168.122.101"

with SimServOven(HOST) as oven:
    oven.enable_manual(True)
    oven.set_temperature(23.0)
    print("Temp real:", oven.get_actual_temperature())
