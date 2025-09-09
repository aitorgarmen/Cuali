# simserv_oven.py
"""Minimal cliente Python para hornos S!MPAC® mediante el protocolo SimServ.

Basado en el manual *Protocolo de comunicación S!MPAC® simserv* (ID doc. es 2017.09 64636633).

El protocolo utiliza *stream sockets TCP* (servidor en el horno, cliente en nuestra
aplicación) y el puerto fijo **2049**. Cada comando es una línea terminada en CR
(ASCII 13) y los campos se separan con el carácter ¶ (ASCII 182, 0xB6).

Este módulo implementa sólo los comandos que necesitamos ahora:

* **11001**  - `SET SETPOINT` : fija el valor nominal de la variable de control 1;
* **14001**  - `START MAN MODE` / `SET DIGITAL OUT` : activamos/desactivamos el
  modo manual (argumento 1 = 1 → ON, 0 → OFF);
* **11004**  - `GET ACTUAL VALUE` : lee el valor real de la variable de control 1.

"""
from __future__ import annotations

import socket
import time
from contextlib import contextmanager
from typing import Optional

DELIM: bytes = b"\xb6"  # \u00B6, ¶
CR: bytes = b"\r"  # Carriage Return


class SimServError(RuntimeError):
    """Error devuelto por el horno o por la conexión."""


class SimServOven:
    """Cliente TCP sencillo para un horno con control S!MPAC®.

    Ejemplo de uso::

        with SimServOven("192.168.121.100") as oven:
            oven.enable_manual(True)
            oven.set_temperature(25.0)
            actual = oven.wait_stable(25.0, hold_s=30, tol=0.2)
            print(f"Estable en {actual:.2f} °C")
    """

    def __init__(self, host: str, port: int = 2049, timeout: float = 10.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None

    # ----------------------------------------------------------------- context
    def __enter__(self) -> "SimServOven":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ----------------------------------------------------------------- básicos
    def connect(self) -> None:
        if self._sock is not None:
            return  # ya conectado
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect((self.host, self.port))
        self._sock = s

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._sock.close()
            self._sock = None

    # ----------------------------------------------------------- helpers internos
    def _build_cmd(self, cmd_id: str, *args: str) -> bytes:
        parts = [cmd_id, "1", *args]  # Índice de cámara se deja siempre en 1
        return DELIM.join(p.encode("ascii") for p in parts) + CR

    def _send_cmd(self, cmd_id: str, *args: str) -> list[str]:
        if self._sock is None:
            self.connect()
        cmd = self._build_cmd(cmd_id, *args)
        assert self._sock is not None  # para mypy
        self._sock.sendall(cmd)
        # Leemos hasta CR (puede venir opcional LF)
        data = bytearray()
        while True:
            chunk = self._sock.recv(512)
            if not chunk:
                raise SimServError("Conexión cerrada inesperadamente")
            data.extend(chunk)
            if CR in chunk:
                break
        # Quitamos CR/LF y decodificamos
        text = data.rstrip(b"\r\n").decode("latin-1")
        return text.split("¶")  # la primera posición es siempre el código de estado

    def _check_ok(self, resp: list[str]) -> None:
        """Levanta `SimServError` si la respuesta indica fallo."""
        if not resp:
            raise SimServError("Respuesta vacía del horno")
        status = resp[0]
        if status != "1":
            raise SimServError(f"Comando rechazado (código {status})  - resto: {resp[1:]}")

    # --------------------------------------------------------- API de alto nivel
    def enable_manual(self, enable: bool = True) -> None:
        """Activa o desactiva el modo manual (necesario para fijar setpoints)."""
        val = "1" if enable else "0"
        resp = self._send_cmd("14001", "1", val)
        self._check_ok(resp)

    def set_temperature(self, setpoint: float) -> None:
        """Envía el setpoint en *°C* a la variable de control nº 1."""
        resp = self._send_cmd("11001", "1", f"{setpoint:.1f}")
        self._check_ok(resp)

    def get_actual_temperature(self) -> float:
        """Lee la temperatura real actual (variable 1)."""
        resp = self._send_cmd("11004", "1")
        self._check_ok(resp)
        if len(resp) < 2:
            raise SimServError(f"Respuesta inesperada: {resp}")
        try:
            return float(resp[1])
        except ValueError as exc:
            raise SimServError(f"Valor no numérico: {resp[1]}") from exc

    def wait_stable(self, target: float, hold_s: int = 30, tol: float = 0.5,
                    poll: float = 1.0, timeout: int | None = 900) -> float:
        """Bloquea hasta que la temperatura esté dentro de `tol` °C durante
        `hold_s` segundos consecutivos (tiempo real).  Devuelve el valor medido.
        """
        start = time.monotonic()
        within = 0
        last = self.get_actual_temperature()
        while True:
            if abs(last - target) <= tol:
                within += poll
                if within >= hold_s:
                    return last
            else:
                within = 0  # reiniciamos contador si sale de tolerancia
            if timeout and (time.monotonic() - start) > timeout:
                raise SimServError("Tiempo de espera agotado estabilizando horno")
            time.sleep(poll)
            last = self.get_actual_temperature()


# --------------------------------------------------------------------------- demo
if __name__ == "__main__":
    import argparse, sys

    p = argparse.ArgumentParser("simserv_oven demo")
    p.add_argument("host", help="IP o hostname del horno")
    p.add_argument("temp", type=float, help="Setpoint en °C")
    p.add_argument("--hold", type=int, default=30, help="Segs dentro de tol para considerar estable")
    p.add_argument("--tol", type=float, default=0.5, help="Tolerancia en °C")
    args = p.parse_args()

    try:
        with SimServOven(args.host) as oven:
            print("Activando modo manual...")
            oven.enable_manual(True)
            print(f"Fijando setpoint a {args.temp:.1f} °C...")
            oven.set_temperature(args.temp)
            print("Esperando estabilidad...")
            real = oven.wait_stable(args.temp, hold_s=args.hold, tol=args.tol)
            print(f"Estable en {real:.2f} °C")
    except SimServError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
