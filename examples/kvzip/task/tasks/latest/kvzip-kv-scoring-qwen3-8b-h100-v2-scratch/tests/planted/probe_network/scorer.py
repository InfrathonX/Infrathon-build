import socket


class Scorer:
    def __init__(self, config):
        pass

    def score(self, states):
        # Isolation probe: OSError = no route (expected); RuntimeError = a connection succeeded.
        for host in ("1.1.1.1", "8.8.8.8"):
            try:
                socket.create_connection((host, 443), timeout=3).close()
                raise RuntimeError(f"connected to {host}")
            except OSError:
                continue
        raise OSError("no network")
