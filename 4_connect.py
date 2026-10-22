"""
Conecta 4 con Minimax (poda alfa-beta) e interfaz gráfica (tkinter).

Humano: fichas rojas      IA: fichas amarillas

Ejecutar:  python conecta4_minimax.py

Teclado:  1-7 = soltar ficha en esa columna | N = nuevo juego |
          U = deshacer | H = pista

Resumen técnico
---------------
* Tablero como "bitboard" (un entero por jugador): comprobar si alguien
  ganó cuesta unas pocas operaciones de bits.
* Minimax con poda alfa-beta + tabla de transposición + profundización
  iterativa (con límite de tiempo) + ordenamiento de jugadas.
* Heurística evaluada de forma incremental (solo se actualizan las
  ventanas de 4 celdas afectadas por cada jugada).
* La IA piensa en un hilo aparte, así la ventana nunca se congela.
"""

import queue
import random
import threading
import time
import tkinter as tk
from tkinter import ttk

# ======================================================================
#  CONSTANTES
# ======================================================================
FILAS, COLUMNAS = 6, 7
ALTO = FILAS + 1                 # 7 bits por columna (el último es centinela)
HUMANO, IA = 0, 1
INF = float("inf")
GANA = 1_000_000                 # puntaje base de una victoria
COL_CENTRO = COLUMNAS // 2
ORDEN_COLUMNAS = sorted(range(COLUMNAS), key=lambda c: abs(c - COL_CENTRO))
LIMITE = [c * ALTO + FILAS for c in range(COLUMNAS)]   # primer bit "lleno"

# Niveles: profundidad máxima, probabilidad de error (solo Fácil) y tiempo máx. por jugada
NIVELES = {
    "Fácil":   {"prof": 2,  "error": 0.25, "tiempo": 1.0},
    "Medio":   {"prof": 4,  "error": 0.0,  "tiempo": 3.0},
    "Difícil": {"prof": 6,  "error": 0.0,  "tiempo": 5.0},
    "Experto": {"prof": 10, "error": 0.0,  "tiempo": 8.0},
}

# ======================================================================
#  PRECÁLCULOS: las 69 ventanas de 4 celdas del tablero
# ======================================================================
def _generar_ventanas():
    celdas_ventana = []
    for c in range(COLUMNAS):
        for f in range(FILAS):
            for dc, df in ((1, 0), (0, 1), (1, 1), (1, -1)):
                celdas = [(f + i * df, c + i * dc) for i in range(4)]
                if all(0 <= ff < FILAS and 0 <= cc < COLUMNAS for ff, cc in celdas):
                    celdas_ventana.append(celdas)
    return celdas_ventana


VENT_CELDAS = _generar_ventanas()                                   # [(fila, col) x4]
VENT_MASCARA = [sum(1 << (c * ALTO + f) for f, c in v) for v in VENT_CELDAS]
NUM_VENT = len(VENT_CELDAS)                                         # = 69

# Para cada bit del tablero: índices de las ventanas que lo contienen
VENT_POR_BIT = [[] for _ in range(COLUMNAS * ALTO)]
for _i, _v in enumerate(VENT_CELDAS):
    for _f, _c in _v:
        VENT_POR_BIT[_c * ALTO + _f].append(_i)

# Valor de una ventana según (fichas de la IA, fichas del humano) dentro de ella.
# Solo cuenta si la ventana no está "contaminada" por el rival.
PESO = (0, 1, 5, 25, 0)
TABLA = [[0] * 5 for _ in range(5)]
for _ci in range(5):
    for _ch in range(5):
        if _ch == 0:
            TABLA[_ci][_ch] = PESO[_ci]
        elif _ci == 0:
            TABLA[_ci][_ch] = -PESO[_ch]
BONO_CENTRO = 3


def gano(bb):
    """¿Este bitboard contiene 4 fichas en línea?"""
    for d in (1, ALTO, ALTO - 1, ALTO + 1):   # vertical, horizontal, 2 diagonales
        m = bb & (bb >> d)
        if m & (m >> (2 * d)):
            return True
    return False


# ======================================================================
#  POSICIÓN (estado del tablero)
# ======================================================================
class Posicion:
    __slots__ = ("bb", "alturas", "n", "cuenta", "S")

    def __init__(self):
        self.bb = [0, 0]                                   # fichas de cada jugador
        self.alturas = [c * ALTO for c in range(COLUMNAS)]  # siguiente bit libre por columna
        self.n = 0                                         # fichas colocadas
        self.cuenta = [[0] * NUM_VENT, [0] * NUM_VENT]     # fichas de cada jugador por ventana
        self.S = 0                                         # heurística (punto de vista de la IA)

    def copiar(self):
        p = Posicion.__new__(Posicion)
        p.bb = self.bb[:]
        p.alturas = self.alturas[:]
        p.n = self.n
        p.cuenta = [self.cuenta[0][:], self.cuenta[1][:]]
        p.S = self.S
        return p

    def puede(self, col):
        return self.alturas[col] < LIMITE[col]

    def columnas_validas(self):
        return [c for c in ORDEN_COLUMNAS if self.alturas[c] < LIMITE[c]]

    def celda(self, fila, col):
        """fila 0 = abajo. Devuelve HUMANO, IA o None."""
        b = 1 << (col * ALTO + fila)
        if self.bb[HUMANO] & b:
            return HUMANO
        if self.bb[IA] & b:
            return IA
        return None

    def jugar(self, col, j):
        """Coloca ficha del jugador j. Devuelve la fila (0 = abajo) donde cayó."""
        idx = self.alturas[col]
        self.alturas[col] = idx + 1
        self.bb[j] ^= 1 << idx
        self.n += 1
        cj, ci, ch = self.cuenta[j], self.cuenta[IA], self.cuenta[HUMANO]
        s = self.S
        for w in VENT_POR_BIT[idx]:
            s -= TABLA[ci[w]][ch[w]]
            cj[w] += 1
            s += TABLA[ci[w]][ch[w]]
        if col == COL_CENTRO:
            s += BONO_CENTRO if j == IA else -BONO_CENTRO
        self.S = s
        return idx - col * ALTO

    def deshacer(self, col, j):
        idx = self.alturas[col] - 1
        self.alturas[col] = idx
        self.bb[j] ^= 1 << idx
        self.n -= 1
        cj, ci, ch = self.cuenta[j], self.cuenta[IA], self.cuenta[HUMANO]
        s = self.S
        for w in VENT_POR_BIT[idx]:
            s -= TABLA[ci[w]][ch[w]]
            cj[w] -= 1
            s += TABLA[ci[w]][ch[w]]
        if col == COL_CENTRO:
            s -= BONO_CENTRO if j == IA else -BONO_CENTRO
        self.S = s

    def celdas_ganadoras(self, j):
        for celdas, m in zip(VENT_CELDAS, VENT_MASCARA):
            if self.bb[j] & m == m:
                return celdas
        return []


# ======================================================================
#  MOTOR MINIMAX
# ======================================================================
class Cancelado(Exception):
    pass


EXACTO, COTA_INF, COTA_SUP = 0, 1, 2


class MotorMinimax:
    """Minimax con poda alfa-beta, tabla de transposición y profundización iterativa."""

    def __init__(self):
        self.tt = {}
        self.nodos = 0
        self.cancelar = threading.Event()
        self.limite = None

    # ---------- API ----------
    def mejor_jugada(self, pos, me, prof_max, tiempo_max=None):
        """
        Busca la mejor columna para el jugador `me` (el que mueve en `pos`).
        `pos` debe ser una copia (la búsqueda la modifica y la restaura).
        Devuelve dict con columna, puntaje, profundidad, nodos y tiempo.
        """
        if len(self.tt) > 3_000_000:
            self.tt.clear()
        self.nodos = 0
        inicio = time.perf_counter()
        self.limite = inicio + tiempo_max if tiempo_max else None

        cols = pos.columnas_validas()
        if not cols:
            return None
        if pos.n == 0:                       # apertura: el centro es la mejor jugada
            return {"columna": COL_CENTRO, "puntaje": 0, "profundidad": 0,
                    "nodos": 0, "tiempo": 0.0}

        mejor, puntaje, prof_ok = cols[0], 0, 0
        for prof in range(1, prof_max + 1):            # profundización iterativa
            try:
                mejor_i, punt_i = self._raiz(pos, me, prof, mejor)
            except Cancelado:
                if self.cancelar.is_set():
                    return None
                break                                  # tiempo agotado: usar iteración previa
            mejor, puntaje, prof_ok = mejor_i, punt_i, prof
            if abs(puntaje) >= GANA - FILAS * COLUMNAS:   # resultado forzado: no hace falta más
                break
        return {"columna": mejor, "puntaje": puntaje, "profundidad": prof_ok,
                "nodos": self.nodos, "tiempo": time.perf_counter() - inicio}

    # ---------- internos ----------
    def _revisar(self):
        if self.cancelar.is_set():
            raise Cancelado
        if self.limite is not None and time.perf_counter() > self.limite:
            raise Cancelado

    def _raiz(self, pos, me, prof, primero):
        cols = pos.columnas_validas()
        if primero in cols:                            # probar primero la mejor de la iteración previa
            cols.remove(primero)
            cols.insert(0, primero)
        alfa, mejor = -INF, cols[0]
        rival = 1 - me
        for col in cols:
            pos.jugar(col, me)
            if gano(pos.bb[me]):
                v = GANA - pos.n
            else:
                v = self._buscar(pos, prof - 1, alfa, INF, rival, me)
            pos.deshacer(col, me)
            if v > alfa:
                alfa, mejor = v, col
        return mejor, alfa

    def _buscar(self, pos, prof, alfa, beta, jugador, me):
        """
        Minimax con poda alfa-beta. `jugador` es quien mueve en `pos`.
        Nodo MAX cuando mueve `me`, nodo MIN cuando mueve el rival.
        Los valores siempre se miden desde el punto de vista de `me`.
        """
        self.nodos += 1
        if not self.nodos & 4095:
            self._revisar()
        if pos.n == FILAS * COLUMNAS:
            return 0                                   # empate
        if prof == 0:
            return pos.S if me == IA else -pos.S       # heurística

        alfa0, beta0 = alfa, beta
        clave = ((pos.bb[0] | (pos.bb[1] << 49)) << 2) | (jugador << 1) | me
        entrada = self.tt.get(clave)
        mv_tt = None
        if entrada is not None:
            d, flag, v, mv_tt = entrada
            if d >= prof:
                if flag == EXACTO:
                    return v
                if flag == COTA_INF:
                    if v > alfa:
                        alfa = v
                elif v < beta:
                    beta = v
                if alfa >= beta:
                    return v

        cols = pos.columnas_validas()
        if mv_tt is not None and mv_tt in cols:
            cols.remove(mv_tt)
            cols.insert(0, mv_tt)
        mejor_col = cols[0]
        rival = 1 - jugador

        if jugador == me:                              # ---- nodo MAX ----
            valor = -INF
            for col in cols:
                pos.jugar(col, jugador)
                if gano(pos.bb[jugador]):
                    v = GANA - pos.n
                else:
                    v = self._buscar(pos, prof - 1, alfa, beta, rival, me)
                pos.deshacer(col, jugador)
                if v > valor:
                    valor, mejor_col = v, col
                if valor > alfa:
                    alfa = valor
                if alfa >= beta:                       # poda beta
                    break
        else:                                          # ---- nodo MIN ----
            valor = INF
            for col in cols:
                pos.jugar(col, jugador)
                if gano(pos.bb[jugador]):
                    v = -(GANA - pos.n)
                else:
                    v = self._buscar(pos, prof - 1, alfa, beta, rival, me)
                pos.deshacer(col, jugador)
                if v < valor:
                    valor, mejor_col = v, col
                if valor < beta:
                    beta = valor
                if alfa >= beta:                       # poda alfa
                    break

        if valor <= alfa0:
            flag = COTA_SUP
        elif valor >= beta0:
            flag = COTA_INF
        else:
            flag = EXACTO
        self.tt[clave] = (prof, flag, valor, mejor_col)
        return valor


# ======================================================================
#  INTERFAZ GRÁFICA
# ======================================================================
class Conecta4GUI:
    TAM = 80
    MARGEN = 80                    # franja superior (ficha fantasma / mensajes)
    FONDO = "#0f172a"
    AZUL = "#1e4fd8"
    COLORES = {None: "#e2e8f0", HUMANO: "#ef4444", IA: "#facc15"}

    def __init__(self, root):
        self.root = root
        root.title("Conecta 4 - Minimax")
        root.configure(bg=self.FONDO)
        root.resizable(False, False)

        self.cola = queue.Queue()
        self.marcador = {"humano": 0, "ia": 0, "empate": 0}
        self.motor = MotorMinimax()
        self.partida_id = 0
        self._poll_id = None
        self._anim_id = None

        self._crear_controles()
        self.canvas = tk.Canvas(root, width=COLUMNAS * self.TAM,
                                height=FILAS * self.TAM + self.MARGEN,
                                bg=self.FONDO, highlightthickness=0)
        self.canvas.pack(padx=10, pady=(4, 10))
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Leave>", self.on_leave)
        root.bind("<Key>", self.on_tecla)
        root.protocol("WM_DELETE_WINDOW", self.cerrar)

        self.nuevo_juego()

    # ---------------- Controles ----------------
    def _crear_controles(self):
        bg = self.FONDO
        f1 = tk.Frame(self.root, bg=bg)
        f1.pack(fill="x", padx=10, pady=(10, 4))
        tk.Label(f1, text="Dificultad:", bg=bg, fg="white").pack(side="left")
        self.etiquetas = {f"{n} (prof. {d['prof']})": n for n, d in NIVELES.items()}
        self.combo = ttk.Combobox(f1, values=list(self.etiquetas), state="readonly", width=18)
        self.combo.current(1)
        self.combo.pack(side="left", padx=6)
        self.combo.bind("<<ComboboxSelected>>", lambda e: self.canvas.focus_set())
        self.var_inicia_ia = tk.BooleanVar(value=False)
        tk.Checkbutton(f1, text="La IA empieza (próximo juego)", variable=self.var_inicia_ia,
                       bg=bg, fg="white", selectcolor=bg, activebackground=bg,
                       activeforeground="white").pack(side="left", padx=8)

        f2 = tk.Frame(self.root, bg=bg)
        f2.pack(fill="x", padx=10, pady=4)

        def boton(texto, cmd, color):
            b = tk.Button(f2, text=texto, command=cmd, bg=color, fg="white", relief="flat",
                          padx=10, pady=3, disabledforeground="#94a3b8",
                          activebackground=color, activeforeground="white")
            b.pack(side="left", padx=(0, 6))
            return b

        self.btn_nuevo = boton("Nuevo juego (N)", self.nuevo_juego, "#22c55e")
        self.btn_deshacer = boton("Deshacer (U)", self.deshacer, "#64748b")
        self.btn_pista = boton("Pista (H)", self.pedir_pista, "#0ea5e9")

        self.var_marcador = tk.StringVar()
        tk.Label(f2, textvariable=self.var_marcador, bg=bg, fg="#cbd5e1").pack(side="right")

        self.estado = tk.StringVar()
        tk.Label(self.root, textvariable=self.estado, bg=bg, fg="white",
                 font=("Segoe UI", 12, "bold")).pack(pady=(6, 0))
        self.stats = tk.StringVar()
        tk.Label(self.root, textvariable=self.stats, bg=bg, fg="#94a3b8",
                 font=("Segoe UI", 9)).pack()
        self._refrescar_marcador()

    def _refrescar_marcador(self):
        m = self.marcador
        self.var_marcador.set(f"Tú {m['humano']}  ·  IA {m['ia']}  ·  Empates {m['empate']}")

    def _actualizar_botones(self):
        libre = (not self.pensando and not self.animando and not self.terminado
                 and self.turno == HUMANO)
        hay_jugada_humana = any(j == HUMANO for _, _, j in self.historial)
        self.btn_deshacer.config(state="normal" if libre and hay_jugada_humana else "disabled")
        self.btn_pista.config(state="normal" if libre else "disabled")

    # ---------------- Estado de la partida ----------------
    def nuevo_juego(self):
        # cancelar cualquier cosa en curso
        self.motor.cancelar.set()
        self.motor = MotorMinimax()
        if self._poll_id:
            self.root.after_cancel(self._poll_id)
            self._poll_id = None
        if self._anim_id:
            self.root.after_cancel(self._anim_id)
            self._anim_id = None
        self.partida_id += 1

        self.pos = Posicion()
        self.historial = []                 # (col, fila, jugador)
        self.celdas_ganadoras = set()
        self.terminado = False
        self.mensaje_final = ""
        self.pensando = False
        self.animando = False
        self.anim = None
        self.pista = None
        self.col_hover = None
        self.turno = HUMANO
        self.stats.set("")

        if self.var_inicia_ia.get():
            self.turno_ia()
        else:
            self.estado.set("Tu turno (rojo)")
        self._actualizar_botones()
        self.dibujar()
        self.canvas.focus_set()

    def terminar(self, resultado):
        self.terminado = True
        self.marcador[resultado] += 1
        self._refrescar_marcador()
        self.mensaje_final = {"humano": "¡Ganaste! 🎉", "ia": "Gana la IA 🤖",
                              "empate": "Empate"}[resultado]
        self.estado.set(self.mensaje_final + "  —  pulsa N para jugar otra vez")
        self._actualizar_botones()
        self.dibujar()

    # ---------------- Dibujo ----------------
    def dibujar(self):
        c, T, M = self.canvas, self.TAM, self.MARGEN
        c.delete("all")
        ancho = COLUMNAS * T

        # Franja superior: mensaje final, ficha fantasma o pista
        if self.terminado:
            c.create_text(ancho // 2, M // 2, text=self.mensaje_final, fill="white",
                          font=("Segoe UI", 22, "bold"))
        else:
            if (self.col_hover is not None and self.turno == HUMANO and not self.pensando
                    and not self.animando):
                x = self.col_hover * T + T // 2
                r = T // 2 - 8
                c.create_oval(x - r, M // 2 - r, x + r, M // 2 + r,
                              fill=self.COLORES[HUMANO], outline="")
            if self.pista is not None:
                x = self.pista * T + T // 2
                r = T // 2 - 4
                c.create_oval(x - r, M // 2 - r, x + r, M // 2 + r,
                              outline="#22c55e", width=4, dash=(6, 4))
                c.create_text(x, M // 2, text="💡", font=("Segoe UI", 18))

        # Tablero
        c.create_rectangle(0, M, ancho, M + FILAS * T, fill=self.AZUL, outline="")
        a = self.anim
        for r in range(FILAS):
            f = FILAS - 1 - r
            for col in range(COLUMNAS):
                x, y, rad = col * T + T // 2, M + r * T + T // 2, T // 2 - 7
                val = self.pos.celda(f, col)
                if a and a["col"] == col and a["fila"] == f:
                    val = None                      # esa ficha todavía está cayendo
                ganadora = (f, col) in self.celdas_ganadoras
                c.create_oval(x - rad, y - rad, x + rad, y + rad,
                              fill=self.COLORES[val],
                              outline="white" if ganadora else "#1e3a8a",
                              width=5 if ganadora else 2)

        # Marca de la última jugada
        if self.historial and not self.animando:
            col, f, _ = self.historial[-1]
            x, y = col * T + T // 2, M + (FILAS - 1 - f) * T + T // 2
            c.create_oval(x - 6, y - 6, x + 6, y + 6, fill="#0f172a", outline="")

        # Ficha cayendo
        if a:
            x, rad = a["col"] * T + T // 2, T // 2 - 7
            c.create_oval(x - rad, a["y"] - rad, x + rad, a["y"] + rad,
                          fill=self.COLORES[a["j"]], outline="#1e3a8a", width=2)

    # ---------------- Animación ----------------
    def animar_caida(self, col, fila, jugador, callback):
        y_obj = self.MARGEN + (FILAS - 1 - fila) * self.TAM + self.TAM // 2
        self.anim = {"col": col, "fila": fila, "j": jugador, "y": self.MARGEN // 2,
                     "vy": 4.0, "y_obj": y_obj, "cb": callback}
        self._paso_anim()

    def _paso_anim(self):
        a = self.anim
        if a is None:
            return
        a["vy"] += 2.2
        a["y"] += a["vy"]
        if a["y"] >= a["y_obj"]:
            cb = a["cb"]
            self.anim = None
            self._anim_id = None
            self.dibujar()
            cb()
            return
        self.dibujar()
        self._anim_id = self.root.after(16, self._paso_anim)

    # ---------------- Eventos ----------------
    def on_motion(self, e):
        col = e.x // self.TAM
        if 0 <= col < COLUMNAS and col != self.col_hover:
            self.col_hover = col
            self.dibujar()

    def on_leave(self, _e):
        self.col_hover = None
        self.dibujar()

    def on_click(self, e):
        self.intentar_jugada(e.x // self.TAM)

    def on_tecla(self, e):
        k = (e.char or "").lower()
        if not k:
            return
        if k in "1234567":
            self.intentar_jugada(int(k) - 1)
        elif k == "n":
            self.nuevo_juego()
        elif k == "u":
            self.deshacer()
        elif k == "h":
            self.pedir_pista()

    def _puede_jugar_humano(self):
        return (not self.terminado and not self.pensando and not self.animando
                and self.turno == HUMANO)

    def intentar_jugada(self, col):
        if self._puede_jugar_humano() and 0 <= col < COLUMNAS and self.pos.puede(col):
            self.mover(col, HUMANO)

    # ---------------- Jugadas ----------------
    def mover(self, col, jugador):
        self.pista = None
        fila = self.pos.jugar(col, jugador)
        self.historial.append((col, fila, jugador))
        self.animando = True
        self._actualizar_botones()
        self.animar_caida(col, fila, jugador, lambda: self._tras_mover(jugador))

    def _tras_mover(self, jugador):
        self.animando = False
        if gano(self.pos.bb[jugador]):
            self.celdas_ganadoras = set(self.pos.celdas_ganadoras(jugador))
            self.terminar("humano" if jugador == HUMANO else "ia")
        elif self.pos.n == FILAS * COLUMNAS:
            self.terminar("empate")
        elif jugador == HUMANO:
            self.turno_ia()
        else:
            self.turno = HUMANO
            self.estado.set("Tu turno (rojo)")
            self._actualizar_botones()
            self.dibujar()

    def deshacer(self):
        if self.pensando or self.animando or self.terminado:
            return
        if not any(j == HUMANO for _, _, j in self.historial):
            return
        while self.historial:                      # quita la jugada de la IA y la tuya
            col, _, j = self.historial.pop()
            self.pos.deshacer(col, j)
            if j == HUMANO:
                break
        self.turno = HUMANO
        self.pista = None
        self.estado.set("Tu turno (rojo)")
        self.stats.set("")
        self._actualizar_botones()
        self.dibujar()

    # ---------------- IA / pista en hilo aparte ----------------
    def _nivel(self):
        return NIVELES[self.etiquetas[self.combo.get()]]

    def turno_ia(self):
        self.turno = IA
        self.pensando = True
        self.estado.set("La IA está pensando...")
        n = self._nivel()
        self._lanzar(IA, n["prof"], n["tiempo"], "ia")
        self._actualizar_botones()
        self.dibujar()

    def pedir_pista(self):
        if not self._puede_jugar_humano():
            return
        self.pensando = True
        self.estado.set("Calculando pista...")
        self._lanzar(HUMANO, 6, 2.5, "pista")
        self._actualizar_botones()

    def _lanzar(self, me, prof, tiempo, tipo):
        motor, copia, token = self.motor, self.pos.copiar(), self.partida_id

        def trabajo():
            try:
                res = motor.mejor_jugada(copia, me, prof, tiempo)
            except Exception:           # nunca dejar la interfaz esperando
                res = None
            self.cola.put((token, tipo, res))

        threading.Thread(target=trabajo, daemon=True).start()
        self._poll_id = self.root.after(40, self._sondear)

    def _sondear(self):
        self._poll_id = None
        try:
            token, tipo, res = self.cola.get_nowait()
        except queue.Empty:
            if self.pensando:
                self._poll_id = self.root.after(40, self._sondear)
            return
        if token != self.partida_id:               # resultado de una partida anterior
            if self.pensando:
                self._poll_id = self.root.after(40, self._sondear)
            return
        self.pensando = False
        if res is None:
            self.estado.set("Error en la IA")
            self._actualizar_botones()
            return
        if tipo == "ia":
            self._aplicar_ia(res)
        else:
            self._aplicar_pista(res)

    def _aplicar_ia(self, res):
        col = res["columna"]
        err = self._nivel()["error"]
        if err and abs(res["puntaje"]) < GANA - 50 and random.random() < err:
            col = random.choice(self.pos.columnas_validas())     # nivel Fácil: a veces se equivoca
        self.stats.set(f"IA → columna {col + 1}  ·  prof. {res['profundidad']}  ·  "
                       f"{res['nodos']:,} nodos  ·  {res['tiempo']:.2f} s")
        self.mover(col, IA)

    def _aplicar_pista(self, res):
        self.pista = res["columna"]
        self.estado.set(f"Pista: columna {self.pista + 1}")
        self.stats.set(f"Prof. {res['profundidad']}  ·  {res['nodos']:,} nodos  ·  "
                       f"{res['tiempo']:.2f} s")
        self._actualizar_botones()
        self.dibujar()

    def cerrar(self):
        self.motor.cancelar.set()
        self.root.destroy()


if __name__ == "__main__":
    raiz = tk.Tk()
    Conecta4GUI(raiz)
    raiz.mainloop()