# device.py
from typing import List, Dict
import serial
import struct
import time
import numpy as np

class Device:
    """
    Encapsula todas las órdenes y el frame de lectura.
    Los métodos que coinciden con los .m devuelven la lista de 10 bytes de eco,
    excepto 'lectura', que devuelve un dict.
    """

    def __init__(
        self,
        port: str | serial.Serial,
        baudrate: int = 115200,
        timeout: float = 0.1,
    ):
        if isinstance(port, serial.Serial):
            self.ser = port
        else:
            self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)

    # --------------------------------------------------------------------- #
    # Escritura genérica de listas de bytes
    # --------------------------------------------------------------------- #
    def _write_seq(self, *vals: int | str) -> None:
        self.ser.write(bytes(_to_uint8(v) for v in vals))

    def _read_echo(self) -> List[str]:
        """Lee los 10 bytes de eco y devuelve ['AA', '0F', ...]."""
        return [f"{b:02X}" for b in _read_exact(self.ser, 10)]

    # --------------------------------------------------------------------- #
    # Comandos "modo ..."
    # --------------------------------------------------------------------- #
    def modo_configure(self) -> List[str]:
        self._write_seq("AA", "02", "0A", "FF",
                        "03", "02", "00", "00", "00", "A3")
        return self._read_echo()

    def modo_assembly(self) -> List[str]:
        self._write_seq("AA", "02", "0A", "FF",
                        "03", "11", "00", "00", "00", "B0")
        return self._read_echo()

    def modo_single(self) -> List[str]:
        self._write_seq("AA", "02", "0A", "FF",
                        "03", "10", "00", "00", "00", "B0")
        return self._read_echo()

    def start_measure(self) -> List[str]:
        # self._write_seq("AA", "02", "0A", "FF",
        #                 "00", "FF", "00", "00", "00", "5D")
        self._write_seq("AA", "02", "0A", "FF",
                        "00", "FF", "01", "00", "00", "5D")
        return self._read_echo()

    def modo_standby(self) -> List[str]:
        self._write_seq("AA", "02", "0A", "FF",
                        "03", "00", "00", "00", "00", "A1")
        return self._read_echo()

    def modo_save(self) -> List[str]:
        self._write_seq("AA", "02", "0A", "FF",
                        "02", "4C", "00", "00", "00", "EC")
        return self._read_echo()

    # --------------------------------------------------------------------- #
    # Comandos de usuario
    # --------------------------------------------------------------------- #
    def enviar(self, dato: int, dir_: str | int) -> List[str]:
        """
        Equivalente a enviar.m
        - dato: entero de 32 bits
        - dir_: byte de dirección (string "10" o int 0x10, por ejemplo)
        """
        dato_bytes_le = dato.to_bytes(4, "little")      # LSB->MSB
        self._write_seq("AA", "02", "0A", "FF", dir_)
        # enviamos en big-endian como hacía MATLAB
        self.ser.write(dato_bytes_le[::-1])
        self._write_seq("D0")
        return self._read_echo()

    def enviar_temp(self) -> List[str]:
        self._write_seq("AA", "02", "0A", "FF",
                        0x08, 0xB8, 0x15, 0x00, 0x00, 0xD0)
        return self._read_echo()

    # --------------------------------------------------------------------- #
    # Lectura de un frame completo (lectura.m)
    # --------------------------------------------------------------------- #
    def lectura(self) -> Dict[str, np.ndarray | float | int]:
        """
        Bloquea hasta recibir la cabecera 0xAA 0x01 0x13 y luego un frame
        completo.  Devuelve un dict:
            dat3, dat2: np.ndarray
            pico1, porcentaje_diferencia
            tof, force, temp
            maxcorrx, maxcorry
        """
        # ── Sincronización ────────────────────────────────────────────────
        sync = (0xAA, 0x01, 0x13)
        pos = 0
        while pos < 3:
            byte = self.ser.read(1)
            if not byte:
                continue          # timeout corto; seguimos esperando
            if byte[0] == sync[pos]:
                pos += 1
            else:
                pos = 1 if byte[0] == sync[0] else 0

        # ── Payload ──────────────────────────────────────────────────────
        _read_exact(self.ser, 3)                           # 3 bytes de relleno
        temp,  = struct.unpack("<f", _read_exact(self.ser, 4))
        tof,   = struct.unpack("<f", _read_exact(self.ser, 4))
        force, = struct.unpack("<f", _read_exact(self.ser, 4))
        _read_exact(self.ser, 1)                           # 1 byte "char"
        maxcorrx, = struct.unpack("<h", _read_exact(self.ser, 2))
        maxcorrx += 1
        maxcorry, = struct.unpack("<i", _read_exact(self.ser, 4))

        dat2 = np.frombuffer(_read_exact(self.ser, 1024 * 2), dtype="<i2")
        dat3 = np.frombuffer(_read_exact(self.ser, 1024 * 4), dtype="<i4")

        # ── Cálculo de picos ─────────────────────────────────────────────
        deriv = np.diff(dat3)
        idxs  = np.where((deriv[:-1] > 0) & (deriv[1:] < 0))[0] + 1  # picos
        pos   = dat3[idxs] > 0
        picos = dat3[idxs][pos]

        if picos.size >= 2:
            orden = np.argsort(picos)[::-1]
            pico1 = int(picos[orden[0]])
            pico2 = int(picos[orden[1]])
            porcentaje = (pico1 - pico2) / pico1 * 100.0
        else:
            pico1, porcentaje = 0, 0.0

        return dict(
            dat3=dat3,
            dat2=dat2,
            pico1=pico1,
            porcentaje_diferencia=porcentaje,
            temp=temp,
            tof=tof,
            force=force,
            maxcorrx=int(maxcorrx),
            maxcorry=int(maxcorry),
        )


def _pack32(value) -> int:
    """
    Devuelve los 32 bits IEEE-754/unsigned equivalentes a `value`.
    Funciona con float, np.float32, int, np.uint32...
    """
    if isinstance(value, (float, np.floating)):
        return np.float32(value).view(np.uint32).item()
    return int(value) & 0xFFFFFFFF

def _to_uint8(value: int | str) -> int:
    """Convierte 'AA', '0xAA' o 170 en un entero 0-255."""
    if isinstance(value, str):
        value = int(value.lower().removeprefix("0x"), 16)
    if not 0 <= value <= 0xFF:
        raise ValueError(f"Fuera de rango uint8: {value}")
    return value


def _read_exact(ser: serial.Serial, n: int) -> bytes:
    """Lee exactamente n bytes o lanza TimeoutError."""
    buf = b""
    deadline = time.monotonic() + (ser.timeout or 0.1) * 2
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if chunk:
            buf += chunk
            deadline = time.monotonic() + (ser.timeout or 0.1) * 2
        elif time.monotonic() > deadline:
            raise TimeoutError(f"Esperando {n} bytes, recibidos {len(buf)}")
    return buf