import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay|overrun_nonfatal;1|fifo_size;50000000"

import socket
import threading
import time
import cv2
import math

# =====================================================================
#  EINSTELLUNGEN
# =====================================================================
DROHNE_IP = "192.168.8.184"
PORT = 8889

FLIEGEN = True        # False = nur testen (Drohne bleibt am Boden), True = fliegen

SCHWELLE = 20          # Grauwert: alles darunter gilt als Linie (im Fenster "Maske" live einstellbar)
MIN_PIXEL = 60        # so viele dunkle Pixel braucht ein Streifen (live einstellbar)

SPEED_VOR = 10          # Vorwaertsgeschwindigkeit (klein anfangen)
DREHUNG = cv2.ROTATE_90_COUNTERCLOCKWISE
RUNTER_NACH_START = 30
KP_SEITE = 0.20        # wie stark seitlich korrigiert wird
KD_SEITE = 0.15        # bremst das Pendeln
KP_DREH = 1         # wie stark in Kurven gedreht wird
MAX_SEITE = 20         # maximale seitliche Geschwindigkeit
MAX_DREH = 30          # maximale Drehgeschwindigkeit
GLAETTUNG = 0.6        # 0 = keine Glaettung, 0.9 = sehr traege

VERLOREN_SEK = 1     # so lange darf die Linie fehlen, dann Landung
MAX_FLUGZEIT = 120     # nach so vielen Sekunden Landung (nur beim Fliegen)
AKKU_MIN_START = 20    # unter diesem Akkustand (%) kein Start
AKKU_MIN_FLUG = 15     # unter diesem Akkustand (%) Landung

# =====================================================================
#  VERBINDUNG
# =====================================================================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock.bind(("", 9000))
except OSError:
    raise SystemExit("Port 9000 ist belegt. Laeuft noch ein altes Skript? "
                     "Alle Terminals und Python-Fenster schliessen.")
sock.settimeout(10)


def sende(befehl):
    sock.sendto(befehl.encode("utf-8"), (DROHNE_IP, PORT))
    try:
        antwort, _ = sock.recvfrom(1024)
        antwort = antwort.decode("utf-8").strip()
    except socket.timeout:
        print(f"{befehl}: FEHLER (keine Antwort)")
        return False
    if antwort.lower() == "ok":
        print(f"{befehl}: geklappt")
        return True
    print(f"{befehl}: Antwort {antwort}")
    return False


def rc(lr, fb, ud, yaw):
    # rc bekommt keine Antwort, deshalb nur senden
    sock.sendto(f"rc {lr} {fb} {ud} {yaw}".encode("utf-8"), (DROHNE_IP, PORT))


def akku():
    # alte Antworten wegraeumen, damit sie sich nicht vermischen
    try:
        sock.settimeout(0.01)
        while True:
            sock.recvfrom(1024)
    except OSError:
        pass
    sock.settimeout(3)
    sock.sendto(b"battery?", (DROHNE_IP, PORT))
    try:
        antwort, _ = sock.recvfrom(1024)
        return int(antwort.decode("utf-8").strip())
    except (OSError, ValueError):
        return None
    finally:
        sock.settimeout(10)


letzter_ping = time.time()


def ping():
    # regelmaessiges Signal, damit die Drohne im Test nicht abschaltet
    global letzter_ping
    if time.time() - letzter_ping > 3:
        rc(0, 0, 0, 0)
        letzter_ping = time.time()


# =====================================================================
#  VIDEO (eigener Thread, damit immer das neueste Bild genutzt wird)
# =====================================================================
class BildLeser:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.bild = None
        self.nummer = 0
        self.laeuft = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._lesen, daemon=True)
        self.thread.start()

    def _lesen(self):
        while self.laeuft:
            ok, b = self.cap.read()
            if ok:
                with self.lock:
                    self.bild = b
                    self.nummer += 1
            else:
                time.sleep(0.01)

    def hole(self):
        with self.lock:
            if self.bild is None:
                return self.nummer, None
            b = self.bild.copy()
            if DREHUNG is not None:
                b = cv2.rotate(b, DREHUNG)
            return self.nummer, b

    def stop(self):
        self.laeuft = False
        self.thread.join(timeout=3)
        self.cap.release()


# =====================================================================
#  HILFSFUNKTIONEN
# =====================================================================
def begrenze(wert, grenze):
    return int(max(-grenze, min(grenze, wert)))


def schwerpunkt_x(maske, y0, y1, min_pixel):
    streifen = maske[y0:y1, :]
    m = cv2.moments(streifen, binaryImage=True)
    if m["m00"] < min_pixel:
        return None
    return m["m10"] / m["m00"]


# =====================================================================
#  HAUPTPROGRAMM
# =====================================================================
leser = None
grund = "unbekannt"
s_aktuell = SCHWELLE
mp_aktuell = MIN_PIXEL

try:
    if not sende("command"):
        raise SystemExit("Drohne nicht erreichbar, Abbruch.")

    stand = akku()
    print("Akku:", stand, "%")
    if FLIEGEN and (stand is None or stand < AKKU_MIN_START):
        raise SystemExit("Akku zu schwach oder keine Antwort, kein Flug.")

    sende("downvision 1")
    sende("streamon")

    print("Warte auf das erste Bild, das kann bis zu 15 Sekunden dauern...")
    leser = BildLeser("udp://0.0.0.0:11111")

    start = time.time()
    nummer, bild = 0, None
    while bild is None:
        if time.time() - start > 40:
            raise SystemExit("Nach 40 Sekunden kein Bild, Abbruch.")
        nummer, bild = leser.hole()
        ping()
        time.sleep(0.05)
    print("Bild kommt an.")

    h, w = bild.shape[:2]
    mitte_x = w / 2
    nah = (int(h * 0.80), int(h * 0.98))    # Streifen direkt unter der Drohne
    fern = (int(h * 0.02), int(h * 0.20))   # Streifen weiter vorne (oben im Bild)

    # Regler fuer Schwelle und Mindestpixel im Fenster "Maske"
    cv2.namedWindow("Maske")
    cv2.createTrackbar("Schwelle", "Maske", SCHWELLE, 255, lambda v: None)
    cv2.createTrackbar("MinPixel", "Maske", MIN_PIXEL, 1000, lambda v: None)

    if FLIEGEN:
        if not sende("takeoff"):
            raise SystemExit("Start fehlgeschlagen, Abbruch.")
        time.sleep(2)
        if RUNTER_NACH_START >= 20:
            sende(f"down {RUNTER_NACH_START}")
            time.sleep(1)

    flug_start = time.time()
    zuletzt_gesehen = time.time()
    letztes_neues_bild = time.time()
    letzte_akkuabfrage = time.time()
    letzte_ausgabe = 0
    letzte_nummer = nummer
    letzter_fehler = 0.0
    fehler_glatt = 0.0
    akku_text = stand

    while True:
        taste = cv2.waitKey(1) & 0xFF
        if taste in (27, ord("q")):
            grund = "ESC oder q gedrueckt"
            break
        if FLIEGEN and time.time() - flug_start > MAX_FLUGZEIT:
            grund = "Maximale Flugzeit erreicht"
            break

        nummer, bild = leser.hole()
        if bild is None or nummer == letzte_nummer:
            if time.time() - letztes_neues_bild > 5:
                grund = "Kein Bild mehr"
                break
            if FLIEGEN and time.time() - letztes_neues_bild > 0.5:
                rc(0, 0, 0, 0)          # kein neues Bild: schweben
            elif not FLIEGEN:
                ping()
            time.sleep(0.005)
            continue
        letzte_nummer = nummer
        letztes_neues_bild = time.time()

        # Akku regelmaessig pruefen
        if time.time() - letzte_akkuabfrage > 15:
            letzte_akkuabfrage = time.time()
            stand = akku()
            if stand is not None:
                akku_text = stand
                print("Akku:", stand, "%")
                if FLIEGEN and stand < AKKU_MIN_FLUG:
                    grund = "Akku fast leer"
                    break

        # Linie erkennen
        s_aktuell = cv2.getTrackbarPos("Schwelle", "Maske")
        mp_aktuell = cv2.getTrackbarPos("MinPixel", "Maske")
        grau = cv2.GaussianBlur(cv2.cvtColor(bild, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        _, maske = cv2.threshold(grau, s_aktuell, 255, cv2.THRESH_BINARY_INV)
        maske = cv2.medianBlur(maske, 5)

        x_nah = schwerpunkt_x(maske, nah[0], nah[1], mp_aktuell)
        x_fern = schwerpunkt_x(maske, fern[0], fern[1], mp_aktuell)

        lr, yaw, vor = 0, 0, 0
        if x_nah is not None or x_fern is not None:
            status = "Linie"
            zuletzt_gesehen = time.time()
            winkel = 0
            if x_nah is not None and x_fern is not None:
                referenz = (x_nah + x_fern) / 2     # Linie auf Höhe der Drohne
                y_nah = (nah[0] + nah[1]) / 2
                y_fern = (fern[0] + fern[1]) / 2
                winkel = math.degrees(math.atan2(x_fern - x_nah, y_nah - y_fern))
                yaw = begrenze(winkel * KP_DREH, MAX_DREH)
            else:
                referenz = x_nah if x_nah is not None else x_fern
            fehler = referenz - mitte_x
            fehler_glatt = GLAETTUNG * fehler_glatt + (1 - GLAETTUNG) * fehler
            lr = begrenze(fehler_glatt * KP_SEITE + (fehler_glatt - letzter_fehler) * KD_SEITE, MAX_SEITE)
            letzter_fehler = fehler_glatt
            # in Kurven und bei großer Abweichung langsamer fliegen
            abweichung = max(min(abs(fehler) / (w / 2), 1), min(abs(winkel) / 45, 1))
            vor = int(SPEED_VOR * (1 - 0.6 * abweichung))
        else:
            x_alle = schwerpunkt_x(maske, 0, h, mp_aktuell)
            if x_alle is not None:
                # Linie nur ausserhalb der Streifen: nur seitlich zur Linie hin
                status = "Linie am Rand"
                zuletzt_gesehen = time.time()
                lr = begrenze((x_alle - mitte_x) * KP_SEITE, MAX_SEITE)
            else:
                status = "KEINE LINIE"
                if time.time() - zuletzt_gesehen > VERLOREN_SEK:
                    grund = "Linie verloren"
                    break

        if FLIEGEN:
            rc(lr, vor, 0, yaw)
        else:
            ping()

        if time.time() - letzte_ausgabe > 1:
            print(f"{status}: nah={x_nah} fern={x_fern} -> seitlich {lr}, vor {vor}, drehen {yaw}")
            letzte_ausgabe = time.time()

        # Anzeige
        for (y0, y1), farbe in ((nah, (0, 255, 0)), (fern, (255, 0, 0))):
            cv2.rectangle(bild, (0, y0), (w - 1, y1), farbe, 1)
        if x_nah is not None:
            cv2.circle(bild, (int(x_nah), (nah[0] + nah[1]) // 2), 5, (0, 255, 0), -1)
        if x_fern is not None:
            cv2.circle(bild, (int(x_fern), (fern[0] + fern[1]) // 2), 5, (255, 0, 0), -1)
        cv2.line(bild, (int(mitte_x), 0), (int(mitte_x), h), (0, 0, 255), 1)
        cv2.putText(bild, f"{status}  Akku {akku_text}%", (4, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.imshow("Kamera", bild)
        cv2.imshow("Maske", maske)

finally:
    print("Ende:", grund)
    print(f"Zuletzt eingestellt: SCHWELLE = {s_aktuell}, MIN_PIXEL = {mp_aktuell}")
    try:
        rc(0, 0, 0, 0)
        if FLIEGEN:
            sende("land")
        sende("streamoff")
    finally:
        if leser is not None:
            leser.stop()
        cv2.destroyAllWindows()