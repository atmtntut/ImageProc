import serial
import time
from io import BytesIO

class GSerial():
    def __init__(self, path, baud, parity='N', stop=1, bytesize=8):
        self.open(path, baud, parity, stop, bytesize)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ser:
            self.ser.close()

    def open(self, path, baud, parity, stop, bytesize):
        try:
            self.ser = serial.Serial(path, baud)
            self.ser.bytesize = bytesize
            self.ser.parity = parity
            self.ser.stopbits = stop
        except Exception as e:
            self.ser = None
            print(e)

    def close(self):
        if self.ser != None:
            self.ser.close()

    def write(self, text):
        if self.ser != None:
            try:
                ret = self.ser.write(text)
                return
            except:
                return ''
        return ''

    def read(self, length=0, timeout=0):
        result = BytesIO()
        offset = 0
        while True:
            time.sleep(0.02)
            timeout -= 0.02
            if self.ser.in_waiting:
                ret = self.ser.read(self.ser.in_waiting)
                offset += result.write(ret)
                if length>0 and offset >= length:
                    break
            if timeout <= 0.0:
                break
        return result.getvalue()

def test():
    with GSerial('/dev/ttyUSB0', 19200, 'E', 2, 8) as ser:
        ser.write(b'1V\r\n')
        print(ser.read(0.05))

    with GSerial('/dev/ttyACM0', 115200, 'N', 1, 8) as ser:
        ser.write(b'VERSION\r')
        print(ser.read(0.02))

if __name__=='__main__':
    test()
