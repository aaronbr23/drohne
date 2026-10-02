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
DROHNE_IP = "10.0.11.109"
PORT = 8889

FLIEGEN = True       # False = nur testen (Drohne bleibt am Boden), True = fliegen

ABSTAND = 40           # so viel dunkler als der Boden muss die Linie sein (im Fenster "Maske" live einstellbar)
MIN_PIXEL = 60        # so viele dunkle Pixel braucht ein Streifen (live einstellbar)

SPEED_VOR = 20          # Vorwaertsgeschwindigkeit (unter ca. 15 bewegt sich die Tello kaum)
DREHUNG = cv2.ROTATE_90_COUNTERCLOCKWISE
SPIEGELN = False       # True, falls links/rechts im Bild vertauscht ist (vorher mit FLIEGEN = False testen!)
KAMERA_HOEHE = 240     # Bodenkamera = 320x240. Die Tello schickt ein 320x720-Bild, nur die oberen 240 Zeilen sind echt.
RUNTER_NACH_START = 30
KP_SEITE = 0.20        # wie stark seitlich korrigiert wird
KD_SEITE = 0.15        # bremst das Pendeln
KP_DREH = 0.8          # wie stark in Kurven gedreht wird (pro Grad Abweichung)
MAX_SEITE = 20         # maximale seitliche Geschwindigkeit
MAX_DREH = 50          # maximale Drehgeschwindigkeit
GLAETTUNG = 0.4        # 0 = keine Glaettung, 0.9 = sehr traege
KURVE_BREMSEN = 0.8    # 0 = in Kurven nicht bremsen, 1 = bei 60 Grad ganz anhalten
SUCH_DREH = 25         # Drehgeschwindigkeit, wenn die Linie kurz weg ist

VERLOREN_SEK = 3     # so lange darf die Linie fehlen, dann Landung
MAX_FLUGZEIT = 120     # nach so vielen Sekunden Landung (nur beim Fliegen)
AKKU_MIN_START = 20    # unter diesem Akkustand (%) kein Start
AKKU_MIN_FLUG = 15     # unter diesem Akkustand (%) Landung

ROHBILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kamerabild_roh.png")

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
        self.url = url
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.bild = None
        self.nummer = 0
        self.laeuft = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._lesen, daemon=True)
        self.thread.start()

    def _lesen(self):
        letztes_ok = time.time()
        while self.laeuft:
            ok, b = self.cap.read()
            if ok:
                letztes_ok = time.time()
                with self.lock:
                    self.bild = b
                    self.nummer += 1
            elif time.time() - letztes_ok > 3:
                # Stream haengt: neu verbinden
                print("Video haengt, verbinde neu...")
                self.cap.release()
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                letztes_ok = time.time()
            else:
                time.sleep(0.01)

    def hole(self):
        with self.lock:
            if self.bild is None:
                return self.nummer, None
            nummer, b = self.nummer, self.bild.copy()
        # nur der echte Teil des Bodenkamera-Bilds, der Rest ist Muell und verschiebt die Bildmitte
        if b.shape[0] > KAMERA_HOEHE:
            b = b[:KAMERA_HOEHE]
        if DREHUNG is not None:
            b = cv2.rotate(b, DREHUNG)
        if SPIEGELN:
            b = cv2.flip(b, 1)
        return nummer, b

    def stop(self):
        self.laeuft = False
        self.thread.join(timeout=3)
        self.cap.release()


# =====================================================================
#  HILFSFUNKTIONEN
# =====================================================================
def begrenze(wert, grenze):
    return int(max(-grenze, min(grenze, wert)))


def linien_maske(grau, abstand):
    # Schwelle relativ zur Bodenhelligkeit, damit Licht und Bodenfarbe egal sind
    boden = float(cv2.medianBlur(grau, 5).mean())
    _, maske = cv2.threshold(grau, max(boden - abstand, 0), 255, cv2.THRESH_BINARY_INV)
    maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    # nur den groessten dunklen Fleck behalten, Schatten und Rauschen fliegen raus
    anzahl, beschriftung, werte, _ = cv2.connectedComponentsWithStats(maske)
    if anzahl <= 1:
        return maske, boden
    groesster = 1 + int(werte[1:, cv2.CC_STAT_AREA].argmax())
    return ((beschriftung == groesster) * 255).astype("uint8"), boden


def schwerpunkt(maske, y0, y1, min_pixel):
    # Mittelpunkt (x, y) der Linie im Streifen y0..y1, oder None
    streifen = maske[y0:y1, :]
    m = cv2.moments(streifen, binaryImage=True)
    if m["m00"] == 0 or m["m00"] < min_pixel:
        return None
    return m["m10"] / m["m00"], y0 + m["m01"] / m["m00"]


# =====================================================================
#  HAUPTPROGRAMM
# =====================================================================
leser = None
grund = "unbekannt"
s_aktuell = ABSTAND
mp_aktuell = MIN_PIXEL

try:
    if not sende("command"):
        raise SystemExit("Drohne nicht erreichbar, Abbruch.")

    stand = akku()
    print("Akku:", stand, "%")
    if FLIEGEN and (stand is None or stand < AKKU_MIN_START):
        raise SystemExit("Akku zu schwach oder keine Antwort, kein Flug.")

    sende("streamoff")                      # haengenden Stream vom letzten Lauf beenden
    time.sleep(0.5)
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
    # die ersten Bilder sind oft noch fehlerhaft, deshalb kurz warten
    start = time.time()
    while time.time() - start < 2:
        nummer, bild = leser.hole()
        ping()
        time.sleep(0.05)

    cv2.imwrite(ROHBILD, bild)
    print("Rohbild gespeichert:", ROHBILD)
    h, w = bild.shape[:2]
    print(f"Kamerabild: {w}x{h} (erwartet 240x320)")
    if FLIEGEN and (w, h) != (240, 320):
        raise SystemExit("Kamerabild hat nicht die erwartete Groesse, bitte erst mit FLIEGEN = False pruefen.")
    mitte_x, mitte_y = w / 2, h / 2         # die Kamera sitzt unter der Bildmitte
    mitte = (int(h * 0.35), int(h * 0.65))  # Streifen auf Hoehe der Drohne -> seitliche Abweichung
    vorne = (0, int(h * 0.35))              # Streifen vor der Drohne -> Zielpunkt fuer die Richtung

    # Regler fuer Schwelle und Mindestpixel im Fenster "Maske"
    cv2.namedWindow("Maske")
    cv2.createTrackbar("Abstand", "Maske", ABSTAND, 255, lambda v: None)
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
    yaw_glatt = 0.0
    letzte_seite = 1                        # in welche Richtung die Linie zuletzt abgebogen ist
    akku_text = stand

    while True:
        taste = cv2.waitKey(1) & 0xFF
        if taste == ord("s"):
            cv2.imwrite(ROHBILD, leser.hole()[1])
            print("Rohbild gespeichert:", ROHBILD)
        if taste in (27, ord("q")):
            grund = "ESC oder q gedrueckt"
            break
        if FLIEGEN and time.time() - flug_start > MAX_FLUGZEIT:
            grund = "Maximale Flugzeit erreicht"
            break

        nummer, bild = leser.hole()
        if bild is None or nummer == letzte_nummer:
            if time.time() - letztes_neues_bild > 10:
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
            if FLIEGEN:
                rc(0, 0, 0, 0)          # akku() blockiert bis 3 s: solange schweben statt blind weiterfliegen
            stand = akku()
            if stand is not None:
                akku_text = stand
                print("Akku:", stand, "%")
                if FLIEGEN and stand < AKKU_MIN_FLUG:
                    grund = "Akku fast leer"
                    break

        # Linie erkennen
        s_aktuell = cv2.getTrackbarPos("Abstand", "Maske")
        mp_aktuell = cv2.getTrackbarPos("MinPixel", "Maske")
        grau = cv2.GaussianBlur(cv2.cvtColor(bild, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        maske, boden = linien_maske(grau, s_aktuell)

        p_mitte = schwerpunkt(maske, mitte[0], mitte[1], mp_aktuell)
        p_vorne = schwerpunkt(maske, vorne[0], vorne[1], mp_aktuell)

        lr, yaw, vor = 0, 0, 0
        if p_mitte is not None or p_vorne is not None:
            status = "Linie"
            zuletzt_gesehen = time.time()
            # Richtung: von der Drohne (Bildmitte) zum Zielpunkt vorne.
            # Faellt die Linie in einer Kurve seitlich aus dem Bild, liegt der Zielpunkt
            # weit aussen und die Drohne dreht kraeftig.
            ziel = p_vorne if p_vorne is not None else p_mitte
            winkel = math.degrees(math.atan2(ziel[0] - mitte_x, max(mitte_y - ziel[1], 1)))
            yaw_glatt = GLAETTUNG * yaw_glatt + (1 - GLAETTUNG) * winkel * KP_DREH
            yaw = begrenze(yaw_glatt, MAX_DREH)
            if abs(winkel) > 10:
                letzte_seite = 1 if winkel > 0 else -1

            # seitlich: Linie unter der Drohne halten
            referenz = p_mitte[0] if p_mitte is not None else ziel[0]
            fehler = referenz - mitte_x
            fehler_glatt = GLAETTUNG * fehler_glatt + (1 - GLAETTUNG) * fehler
            lr = begrenze(fehler_glatt * KP_SEITE + (fehler_glatt - letzter_fehler) * KD_SEITE, MAX_SEITE)
            letzter_fehler = fehler_glatt

            # in Kurven langsamer, damit die Drehung hinterherkommt
            vor = int(SPEED_VOR * (1 - KURVE_BREMSEN * min(abs(winkel) / 60, 1)))
            if p_vorne is None:
                vor = vor // 2                  # vorne nichts zu sehen (Kurve/Ende): vorsichtig
        else:
            fehler_glatt = letzter_fehler = 0.0     # alter Zustand wuerde beim Wiederfinden einen D-Sprung ausloesen
            p_alle = schwerpunkt(maske, 0, h, mp_aktuell)
            if p_alle is not None:
                # Linie nur hinter der Drohne: dorthin zurueck
                status = "Linie hinten"
                zuletzt_gesehen = time.time()
                lr = begrenze((p_alle[0] - mitte_x) * KP_SEITE, MAX_SEITE)
                vor = begrenze((mitte_y - p_alle[1]) * KP_SEITE, MAX_SEITE)
            else:
                # kurz weg: auf der Stelle in die letzte Kurvenrichtung drehen und suchen
                status = "SUCHE"
                yaw = SUCH_DREH * letzte_seite
                if time.time() - zuletzt_gesehen > VERLOREN_SEK:
                    grund = "Linie verloren"
                    break

        if FLIEGEN:
            rc(lr, vor, 0, yaw)
        else:
            ping()

        if time.time() - letzte_ausgabe > 1:
            print(f"{status} (Boden {boden:.0f}, Bildmitte x={mitte_x:.0f}): mitte={p_mitte} vorne={p_vorne} -> seitlich {lr}, vor {vor}, drehen {yaw}")
            letzte_ausgabe = time.time()

        # Anzeige
        for (y0, y1), farbe in ((mitte, (0, 255, 0)), (vorne, (255, 0, 0))):
            cv2.rectangle(bild, (0, y0), (w - 1, y1), farbe, 1)
        for p, farbe in ((p_mitte, (0, 255, 0)), (p_vorne, (255, 0, 0))):
            if p is not None:
                cv2.circle(bild, (int(p[0]), int(p[1])), 5, farbe, -1)
        cv2.line(bild, (int(mitte_x), 0), (int(mitte_x), h), (0, 0, 255), 1)
        # Pfeil: wohin die Drohne gerade fliegen will (oben = vorwaerts, rechts = rechts)
        cv2.arrowedLine(bild, (w // 2, h // 2), (w // 2 + lr * 3, h // 2 - vor * 3), (0, 0, 255), 2)
        cv2.putText(bild, f"dreh {yaw:+d}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(bild, f"{status}  Akku {akku_text}%", (4, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.imshow("Kamera", bild)
        cv2.imshow("Maske", maske)

finally:
    print("Ende:", grund)
    print(f"Zuletzt eingestellt: ABSTAND = {s_aktuell}, MIN_PIXEL = {mp_aktuell}")
    try:
        rc(0, 0, 0, 0)
        if FLIEGEN:
            sende("land")
        sende("streamoff")
    finally:
        if leser is not None:
            leser.stop()
        cv2.destroyAllWindows()