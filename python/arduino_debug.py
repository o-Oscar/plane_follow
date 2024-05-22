import serial
import time

"""
L'arduino transmet quelques messages à la caméra pour initialiser la communication.
Ensuite, l'arduino se met en mode transparent et laisse passer les messages entre l'ordinateur et la caméra sans y toucher.

Dans ce script, des commandes simples sont envoyées pour vérifier que l'arduino fait bien son taff et qu'on est bien capable de piloter la caméra.
"""

arduino = serial.Serial(port="/dev/ttyACM0", baudrate=38400, timeout=0.1)


while True:

    print("press to start sequence")
    input()

    print("0x" + " 0x".join(format(x, "02x") for x in arduino.read_all()))

    if True:
        # custom command
        arduino.write(
            bytearray([0xAA, 0x08, 0x08, 0x02, 0x03, 0x00, 0x00, 0x00, 0x00, 0xFF])
        )

        time.sleep(1)
        print("0x" + " 0x".join(format(x, "02x") for x in arduino.read_all()))

        # stop
        arduino.write(bytearray([0x81, 0x01, 0x06, 0x01, 0x08, 0x08, 0x03, 0x03, 0xFF]))

        time.sleep(0.3)
        print("0x" + " 0x".join(format(x, "02x") for x in arduino.read_all()))

        # home
        arduino.write(bytearray([0x81, 0x01, 0x06, 0x04, 0xFF]))

        time.sleep(0.3)
        print("0x" + " 0x".join(format(x, "02x") for x in arduino.read_all()))

    else:
        # move right
        arduino.write(bytearray([0x81, 0x01, 0x06, 0x01, 0x08, 0x08, 0x02, 0x03, 0xFF]))

        time.sleep(0.6)
        print("0x" + " 0x".join(format(x, "02x") for x in arduino.read_all()))

        # move right
        arduino.write(bytearray([0x81, 0x01, 0x06, 0x01, 0x0A, 0x0A, 0x02, 0x03, 0xFF]))

        time.sleep(0.6)
        print("0x" + " 0x".join(format(x, "02x") for x in arduino.read_all()))

        # move right
        arduino.write(bytearray([0x81, 0x01, 0x06, 0x01, 0x02, 0x02, 0x02, 0x03, 0xFF]))

        time.sleep(0.6)
        print("0x" + " 0x".join(format(x, "02x") for x in arduino.read_all()))

        # stop
        arduino.write(bytearray([0x81, 0x01, 0x06, 0x01, 0x08, 0x08, 0x03, 0x03, 0xFF]))

        time.sleep(0.3)
        print("0x" + " 0x".join(format(x, "02x") for x in arduino.read_all()))

        # home
        arduino.write(bytearray([0x81, 0x01, 0x06, 0x04, 0xFF]))

        time.sleep(0.3)
        print("0x" + " 0x".join(format(x, "02x") for x in arduino.read_all()))
