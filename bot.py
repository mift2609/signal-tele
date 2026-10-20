import asyncio
import json
import logging
import os
import uuid

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Semua rahasia dibaca dari file .env (JANGAN ditulis di dalam file ini, repo GitHub bersifat publik).
TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
AI_API_KEY = os.environ["AI_API_KEY"]
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("AI_MODEL", "qwen-plus")
# WAJIB diisi (ID Telegram dipisah koma) supaya hanya Anda yang bisa memakai bot.
ALLOWED = {int(x) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()}

LOT = float(os.getenv("LOT_SIZE", "0.01"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
SYMBOL_SUFFIX = os.getenv("SYMBOL_SUFFIX", "")  # contoh "-STDc" kalau simbol broker XAUUSD-STDc
ALLOW_REAL = os.getenv("ALLOW_REAL", "0") == "1"  # default: eksekusi hanya di akun DEMO
EXEC_ENABLED = bool(ALLOWED)
MAGIC = 20261020

# Sinyal BUY/SELL hanya keluar kalau skor konfluensi >= MIN_SCORE. Target reward:risk = RR.
MIN_SCORE = int(os.getenv("MIN_SCORE", "50"))
RR = float(os.getenv("RR", "2.0"))

# Backtest: jumlah candle yang diuji dan batas lama posisi (dalam candle) sebelum ditutup paksa
BT_BARS = int(os.getenv("BT_BARS", "2000"))
BT_MAX_HOLD = int(os.getenv("BT_MAX_HOLD", "100"))
PENDING_BARS = int(os.getenv("PENDING_BARS", "12"))  # order LIMIT/STOP dianggap batal bila tak terisi dalam N candle
TF_MIN = {"1min": 1, "5min": 5, "15min": 15, "30min": 30, "1h": 60, "4h": 240, "1day": 1440}
HTF_MIN = {"1min": 15, "5min": 60, "15min": 240, "30min": 240, "1h": 240, "4h": 1440, "1day": 10080}

# Pair yang muncul di /menu (bisa diubah lewat MENU_PAIRS di .env, dipisah koma)
MENU_PAIRS = [p.strip().upper() for p in os.getenv("MENU_PAIRS", "XAUUSD,EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD").split(",") if p.strip()]
MENU_TFS = ["5min", "15min", "30min", "1h", "4h"]

ai = AsyncOpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)
DEFAULT_TF = "1h"
TF_MAP = {
    "1min": mt5.TIMEFRAME_M1,
    "5min": mt5.TIMEFRAME_M5,
    "15min": mt5.TIMEFRAME_M15,
    "30min": mt5.TIMEFRAME_M30,
    "1h": mt5.TIMEFRAME_H1,
    "4h": mt5.TIMEFRAME_H4,
    "1day": mt5.TIMEFRAME_D1,
}
# Timeframe yang lebih besar untuk konfirmasi arah tren (higher timeframe)
HTF_MAP = {
    "1min": ("15min", mt5.TIMEFRAME_M15),
    "5min": ("1h", mt5.TIMEFRAME_H1),
    "15min": ("4h", mt5.TIMEFRAME_H4),
    "30min": ("4h", mt5.TIMEFRAME_H4),
    "1h": ("4h", mt5.TIMEFRAME_H4),
    "4h": ("1day", mt5.TIMEFRAME_D1),
    "1day": ("1minggu", mt5.TIMEFRAME_W1),
}
TF_ALIAS = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "60m": "1h", "1d": "1day", "d1": "1day", "h1": "1h", "h4": "4h"}

EXPLAIN_PROMPT = (
    "Kamu analis forex profesional. Tulis penjelasan singkat (maksimal 3 kalimat, Bahasa Indonesia, "
    "tanpa markdown) atas keputusan sistem berikut. JANGAN mengubah keputusan, arah, atau angka. "
    "Sebutkan satu risiko utama."
)


async def run_sync(func, *args):
    """Jalankan fungsi sinkron di thread terpisah (kompatibel Python 3.8)."""
    return await asyncio.get_running_loop().run_in_executor(None, func, *args)


def normalize_pair(raw: str) -> str:
    return raw.upper().replace("/", "").replace("-", "")


# ---------- MT5 (fungsi sinkron, dipanggil lewat run_sync) ----------
def ensure_mt5():
    if mt5.terminal_info() is None and not mt5.initialize():
        raise RuntimeError(f"MT5 tidak terhubung: {mt5.last_error()}. Pastikan terminal MT5 terbuka dan sudah login.")


def resolve_symbol(pair: str) -> str:
    ensure_mt5()
    sym = pair + SYMBOL_SUFFIX
    if not mt5.symbol_select(sym, True):
        found = [s.name for s in (mt5.symbols_get(f"*{pair}*") or [])][:5]
        hint = f" Mungkin maksud: {', '.join(found)}" if found else ""
        raise ValueError(f"Simbol {sym} tidak ditemukan di broker.{hint}")
    return sym


def rates_df(sym: str, tf_const, n: int, min_len: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(sym, tf_const, 0, n)
    if rates is None or len(rates) < min_len:
        raise ValueError(
            f"Data candle belum cukup (butuh minimal {min_len}). Buka grafik {sym} di MT5 dulu agar riwayat terunduh."
        )
    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    return df.iloc[:-1].reset_index(drop=True)  # buang candle yang masih berjalan, pakai candle tertutup saja


def collect_market_data(pair: str, tf: str) -> dict:
    sym = resolve_symbol(pair)
    info = mt5.symbol_info(sym)
    tick = mt5.symbol_info_tick(sym)
    df = rates_df(sym, TF_MAP[tf], 300, 120)
    htf_name, htf_const = HTF_MAP[tf]
    hdf = rates_df(sym, htf_const, 300, 60)
    if tick is not None and tick.ask and tick.bid:
        ask, bid = float(tick.ask), float(tick.bid)
    else:
        ask = bid = float(df["close"].iloc[-1])
    digits = int(info.digits) if info is not None else 5
    return {"df": df, "htf": hdf, "htf_name": htf_name, "ask": ask, "bid": bid, "digits": digits}


# ---------- Mesin analisis (murni perhitungan, tanpa AI) ----------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.where(loss != 0)
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - c.shift()).abs(), (df["low"] - c.shift()).abs()],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    return df


def htf_bias(h: pd.DataFrame) -> str:
    h = add_indicators(h.copy())
    close = float(h["close"].iloc[-1])
    e50 = float(h["ema50"].iloc[-1])
    e200 = float(h["ema200"].iloc[-1])
    slope = float(h["ema50"].iloc[-1] - h["ema50"].iloc[-6])
    if close > e50 > e200 and slope > 0:
        return "BULL"
    if close < e50 < e200 and slope < 0:
        return "BEAR"
    return "NEUTRAL"


def swing_levels(df: pd.DataFrame, k: int = 2, lookback: int = 120):
    d = df.tail(lookback)
    hi = d["high"].to_numpy()
    lo = d["low"].to_numpy()
    highs, lows = [], []
    for i in range(k, len(d) - k):
        if all(hi[i] > hi[i - j] and hi[i] >= hi[i + j] for j in range(1, k + 1)):
            highs.append(float(hi[i]))
        if all(lo[i] < lo[i - j] and lo[i] <= lo[i + j] for j in range(1, k + 1)):
            lows.append(float(lo[i]))
    return highs, lows


def evaluate(df: pd.DataFrame, hdf: pd.DataFrame, htf_name: str, ask: float, bid: float, digits: int) -> dict:
    """Hitung skor konfluensi (0-100) untuk BUY dan SELL, lalu pilih yang terbaik."""
    d = add_indicators(df.copy())
    last, prev = d.iloc[-1], d.iloc[-2]
    atr = float(last["atr"])
    if not atr > 0:
        raise ValueError("ATR tidak valid (data harga datar)")
    rsi = float(last["rsi"])
    e20, e50, e200 = float(last["ema20"]), float(last["ema50"]), float(last["ema200"])
    hist, hist_prev = float(last["hist"]), float(prev["hist"])
    bias = htf_bias(hdf)
    mid = (ask + bid) / 2
    spread = ask - bid
    highs, lows = swing_levels(d)

    results = {}
    for side in ("BUY", "SELL"):
        buy = side == "BUY"
        sgn = 1 if buy else -1
        entry = ask if buy else bid
        f = []

        # 1) Arah tren timeframe besar
        want = "BULL" if buy else "BEAR"
        f.append((f"Tren HTF {htf_name}", 25 if bias == want else (8 if bias == "NEUTRAL" else 0), 25))

        # 2) Susunan EMA 20/50/200
        full = (e20 > e50 > e200) if buy else (e20 < e50 < e200)
        half = (e20 > e50) if buy else (e20 < e50)
        f.append(("Tren EMA", 20 if full else (10 if half else 0), 20))

        # 3) Momentum MACD
        h_s, hp_s = hist * sgn, hist_prev * sgn
        f.append(("MACD", 10 if (h_s > 0 and h_s > hp_s) else (5 if h_s > 0 else 0), 10))

        # 4) RSI (hindari overbought/oversold)
        if buy:
            p = 10 if 45 <= rsi <= 65 else (5 if 65 < rsi <= 70 else 0)
        else:
            p = 10 if 35 <= rsi <= 55 else (5 if 30 <= rsi < 35 else 0)
        f.append(("RSI", p, 10))

        # 5) Lokasi entry: dekat EMA20 (pullback), bukan mengejar harga
        right_side = (mid > e50) if buy else (mid < e50)
        dist = abs(mid - e20) / atr
        p = (15 if dist <= 0.5 else (8 if dist <= 1.0 else 0)) if right_side else 0

        # Jenis order terbaik: MARKET (harga sudah bagus), LIMIT (tunggu pullback ke EMA20
        # kalau harga sudah menjauh), STOP (tunggu tembus swing terdekat bila momentum kuat)
        tipe, px = "MARKET", entry
        pull_ok = (e20 < entry) if buy else (e20 > entry)
        if p == 0 and right_side and pull_ok:
            tipe, px, p = "LIMIT", e20, 8  # setup valid, tapi masuk di EMA20 bukan mengejar harga
        f.append(("Posisi entry", p, 15))
        if tipe == "MARKET" and f[2][1] == 10:
            if buy:
                near = [x for x in highs if entry + 0.1 * atr <= x <= entry + 0.6 * atr]
                if near:
                    tipe, px = "STOP", min(near) + 0.1 * atr
            else:
                near = [x for x in lows if entry - 0.6 * atr <= x <= entry - 0.1 * atr]
                if near:
                    tipe, px = "STOP", max(near) - 0.1 * atr

        # Level: SL di bawah/atas swing terdekat (atau 1.5 ATR), TP = RR x risiko
        risk = 1.5 * atr
        if buy:
            below = [x for x in lows if x < px]
            if below:
                s_sl = max(below) - 0.25 * atr
                if atr <= px - s_sl <= 3.0 * atr:
                    risk = px - s_sl
        else:
            above = [x for x in highs if x > px]
            if above:
                s_sl = min(above) + 0.25 * atr
                if atr <= s_sl - px <= 3.0 * atr:
                    risk = s_sl - px
        sl = px - sgn * risk
        tp = px + sgn * RR * risk

        # 6) Ruang menuju TP: tidak boleh terhalang support/resistance terdekat
        if buy:
            opp = [x for x in highs if x > px + 0.1 * atr]
            nearest = min(opp) if opp else None
        else:
            opp = [x for x in lows if x < px - 0.1 * atr]
            nearest = max(opp) if opp else None
        room = 99.0 if nearest is None else abs(nearest - px) / atr
        need = RR * risk / atr
        f.append(("Ruang S/R", 10 if room >= need else (4 if room >= 0.6 * need else 0), 10))

        # 7) Spread terhadap volatilitas
        ratio = spread / atr
        f.append(("Spread", 10 if ratio <= 0.15 else (5 if ratio <= 0.3 else 0), 10))

        results[side] = {"score": sum(p for _, p, _ in f), "factors": f, "entry": px, "sl": sl, "tp": tp, "tipe": tipe, "now": entry}

    best = max(results, key=lambda k: results[k]["score"])
    r = results[best]
    blocks = []
    if r["factors"][0][1] == 0:
        blocks.append(f"melawan tren {htf_name}")
    if r["factors"][3][1] == 0:
        blocks.append("RSI di zona ekstrem")
    if r["factors"][4][1] == 0:
        blocks.append("posisi harga tidak ideal (sisi salah EMA50 / terlalu jauh dari EMA20)")
    if r["factors"][6][1] == 0:
        blocks.append("spread terlalu lebar")
    if r["score"] < MIN_SCORE:
        blocks.append(f"skor {r['score']} di bawah {MIN_SCORE}")
    ok = not blocks
    return {
        "arah": best if ok else "WAIT",
        "tipe": r["tipe"] if ok else None,
        "harga_now": round(r["now"], digits),
        "arah_terbaik": best,
        "skor": r["score"],
        "min_skor": MIN_SCORE,
        "faktor": r["factors"],
        "blokir": blocks,
        "htf": htf_name,
        "bias": bias,
        "rsi": round(rsi, 1),
        "atr": round(atr, digits),
        "entry": round(r["entry"], digits) if ok else None,
        "stop_loss": round(r["sl"], digits) if ok else None,
        "take_profit": round(r["tp"], digits) if ok else None,
    }


def fallback_reason(s: dict) -> str:
    kuat = [n for n, p, m in s["faktor"] if p == m]
    lemah = [n for n, p, m in s["faktor"] if p == 0]
    txt = f"Faktor kuat: {', '.join(kuat) or '-'}. Faktor lemah: {', '.join(lemah) or '-'}."
    if s["blokir"]:
        txt += " Ditahan karena " + "; ".join(s["blokir"]) + "."
    return txt


async def explain(pair: str, tf: str, s: dict) -> str:
    """AI hanya menjelaskan keputusan, tidak menentukan arah atau level."""
    fakta = "\n".join(f"- {n}: {p}/{m}" for n, p, m in s["faktor"])
    ringkas = (
        f"Pair {pair}, timeframe {tf}, HTF {s['htf']} bias {s['bias']}, RSI {s['rsi']}, ATR {s['atr']}.\n"
        f"Keputusan sistem: {s['arah']} (arah terbaik {s['arah_terbaik']}, skor {s['skor']}/100, minimal {s['min_skor']}).\n"
        f"Faktor skor:\n{fakta}\n"
    )
    if s["arah"] != "WAIT":
        ringkas += f"Jenis order {s['tipe']}, entry {s['entry']} (harga kini {s['harga_now']}), SL {s['stop_loss']}, TP {s['take_profit']}.\n"
    if s["blokir"]:
        ringkas += "Alasan WAIT: " + "; ".join(s["blokir"]) + "\n"
    try:
        resp = await ai.chat.completions.create(
            model=MODEL,
            max_tokens=300,
            messages=[{"role": "system", "content": EXPLAIN_PROMPT}, {"role": "user", "content": ringkas}],
        )
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return text
    except Exception:
        logging.exception("penjelasan AI gagal, pakai penjelasan otomatis")
    return fallback_reason(s)


async def analyze(pair: str, tf: str) -> dict:
    data = await run_sync(collect_market_data, pair, tf)
    s = evaluate(data["df"], data["htf"], data["htf_name"], data["ask"], data["bid"], data["digits"])
    s["alasan"] = await explain(pair, tf, s)
    return s


def valid_levels(s: dict) -> bool:
    try:
        sl, tp = float(s["stop_loss"]), float(s["take_profit"])
    except (TypeError, ValueError, KeyError):
        return False
    return (s["arah"] == "BUY" and sl < tp) or (s["arah"] == "SELL" and sl > tp)


def execute_order_sync(pair: str, s: dict):
    if not valid_levels(s):
        raise ValueError("Level SL/TP tidak valid")
    ensure_mt5()
    acc = mt5.account_info()
    if acc is None:
        raise RuntimeError("Akun MT5 belum login")
    if acc.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO and not ALLOW_REAL:
        raise ValueError("Akun ini bukan DEMO, eksekusi diblokir (ALLOW_REAL=1 untuk mengizinkan).")
    if len(mt5.positions_get() or ()) + len(mt5.orders_get() or ()) >= MAX_POSITIONS:
        raise ValueError(f"Batas posisi + order tertunda ({MAX_POSITIONS}) tercapai")

    sym = resolve_symbol(pair)
    info = mt5.symbol_info(sym)
    tick = mt5.symbol_info_tick(sym)
    if info is None or tick is None:
        raise RuntimeError("Info simbol tidak tersedia")

    buy = s["arah"] == "BUY"
    price = tick.ask if buy else tick.bid
    tipe = s.get("tipe") or "MARKET"
    sl, tp = float(s["stop_loss"]), float(s["take_profit"])
    entry = float(s["entry"])
    if tipe == "MARKET":
        ok = (sl < price < tp) if buy else (tp < price < sl)
        if not ok:
            raise ValueError(f"SL/TP tidak sesuai harga saat ini ({price}). Minta sinyal baru.")
    else:
        # Order tertunda: harga pasang harus di sisi yang benar dan tidak terlalu dekat dengan harga sekarang
        ok = (entry < price if buy else entry > price) if tipe == "LIMIT" else (entry > price if buy else entry < price)
        if not ok:
            raise ValueError(f"Harga {tipe} {entry} sudah terlewat (harga kini {price}). Minta sinyal baru.")
        min_gap = (info.trade_stops_level or 0) * info.point
        if abs(entry - price) < min_gap:
            raise ValueError(f"Harga {tipe} terlalu dekat dengan harga kini (minimal jarak broker {min_gap}). Minta sinyal baru.")

    fm = info.filling_mode
    if fm & 1:
        filling = mt5.ORDER_FILLING_FOK
    elif fm & 2:
        filling = mt5.ORDER_FILLING_IOC
    else:
        filling = mt5.ORDER_FILLING_RETURN

    if tipe == "MARKET":
        req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": sym,
            "volume": max(LOT, info.volume_min),
            "type": mt5.ORDER_TYPE_BUY if buy else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": MAGIC,
            "comment": "tele-signal-bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
    else:
        otype = {
            ("LIMIT", True): mt5.ORDER_TYPE_BUY_LIMIT,
            ("LIMIT", False): mt5.ORDER_TYPE_SELL_LIMIT,
            ("STOP", True): mt5.ORDER_TYPE_BUY_STOP,
            ("STOP", False): mt5.ORDER_TYPE_SELL_STOP,
        }[(tipe, buy)]
        req = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": sym,
            "volume": max(LOT, info.volume_min),
            "type": otype,
            "price": entry,
            "sl": sl,
            "tp": tp,
            "magic": MAGIC,
            "comment": "tele-signal-bot",
            "type_time": mt5.ORDER_TIME_DAY,  # kedaluwarsa otomatis di akhir hari broker
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
    res = mt5.order_send(req)
    if res is None:
        raise RuntimeError(f"order_send gagal: {mt5.last_error()}")
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        raise RuntimeError(f"Ditolak broker (kode {res.retcode}): {res.comment}")
    return res


def account_text_sync() -> str:
    ensure_mt5()
    a = mt5.account_info()
    if a is None:
        raise RuntimeError("Akun MT5 belum login")
    is_demo = a.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
    pos = len(mt5.positions_get() or ())
    text = f"Akun {a.login} ({'DEMO' if is_demo else 'REAL'})\nServer: {a.server}\nSaldo: {a.balance} {a.currency}\nEkuitas: {a.equity}\nPosisi terbuka: {pos}"
    if not is_demo and not ALLOW_REAL:
        text += "\n\n🔒 Akun REAL: tombol eksekusi order diblokir (hanya sinyal)."
    return text


# ---------- Backtest ----------
def simulate_trade(buy, entry, sl, tp, spread, hi, lo, cl, i, max_hold):
    """Simulasi posisi yang dibuka di akhir candle i.
    Mengembalikan (hasil, indeks_keluar, R) atau None bila belum selesai sampai data habis.
    Kalau SL dan TP tersentuh di candle yang sama, SL dianggap kena lebih dulu (konservatif)."""
    n = len(cl)
    risk = abs(entry - sl)
    last_j = min(n - 1, i + max_hold)
    for j in range(i + 1, last_j + 1):
        if buy:  # BUY keluar di harga bid
            hit_sl, hit_tp = lo[j] <= sl, hi[j] >= tp
        else:  # SELL keluar di harga ask (= bid + spread)
            hit_sl, hit_tp = hi[j] + spread >= sl, lo[j] + spread <= tp
        if hit_sl:
            return "SL", j, -1.0
        if hit_tp:
            return "TP", j, abs(tp - entry) / risk
    if i + max_hold > n - 1:
        return None
    price = float(cl[last_j])
    r = ((price - entry) if buy else (entry - (price + spread))) / risk
    return "TIMEOUT", last_j, r


def summarize_trades(trades: list) -> dict:
    rs = [t["r"] for t in trades]
    n = len(rs)
    gross_win = sum(x for x in rs if x > 0)
    gross_loss = -sum(x for x in rs if x <= 0)
    eq = np.concatenate([[0.0], np.cumsum(rs)]) if n else np.array([0.0])
    streak = best = 0
    for x in rs:
        streak = streak + 1 if x <= 0 else 0
        best = max(best, streak)

    def side_stat(side):
        sub = [t["r"] for t in trades if t["side"] == side]
        return len(sub), (100.0 * sum(1 for x in sub if x > 0) / len(sub) if sub else 0.0)

    return {
        "n": n,
        "tp": sum(1 for t in trades if t["hasil"] == "TP"),
        "sl": sum(1 for t in trades if t["hasil"] == "SL"),
        "timeout": sum(1 for t in trades if t["hasil"] == "TIMEOUT"),
        "win_rate": 100.0 * sum(1 for x in rs if x > 0) / n if n else 0.0,
        "avg_r": float(np.mean(rs)) if n else 0.0,
        "total_r": float(sum(rs)),
        "pf": (gross_win / gross_loss) if gross_loss > 0 else None,
        "max_dd": float((np.maximum.accumulate(eq) - eq).max()),
        "max_streak": best,
        "buy": side_stat("BUY"),
        "sell": side_stat("SELL"),
        "gross_win": gross_win,
    }


def run_backtest(df: pd.DataFrame, hdf: pd.DataFrame, htf_name: str, tf: str, spread: float, digits: int,
                 max_hold: int = 100, start: int = 299) -> dict:
    """Jalankan aturan yang sama dengan sinyal live pada riwayat. Keputusan di candle i hanya memakai data sampai i."""
    n = len(df)
    tsec = TF_MIN[tf] * 60
    hsec = HTF_MIN[tf] * 60
    h_close = hdf["time"].to_numpy() + hsec  # waktu penutupan tiap candle HTF
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    cl = df["close"].to_numpy()
    tm = df["time"].to_numpy()
    trades = []
    unfilled = 0
    i = start
    while i < n - 1:
        k = int(np.searchsorted(h_close, tm[i] + tsec, side="right"))  # hanya candle HTF yang sudah tertutup
        if k < 60:
            i += 1
            continue
        price = float(cl[i])  # harga bid; ask = bid + spread
        try:
            s = evaluate(df.iloc[i - 299 : i + 1], hdf.iloc[max(0, k - 300) : k], htf_name, price + spread, price, digits)
        except ValueError:
            i += 1
            continue
        if s["arah"] == "WAIT":
            i += 1
            continue
        buy = s["arah"] == "BUY"
        tipe = s["tipe"]
        start_i = i
        if tipe != "MARKET":
            # Order tertunda: tunggu terisi maksimal PENDING_BARS candle, kalau tidak terisi dibatalkan
            ep = s["entry"]
            fill = None
            for j in range(i + 1, min(n, i + 1 + PENDING_BARS)):
                if tipe == "LIMIT":
                    hit = (lo[j] + spread <= ep) if buy else (hi[j] >= ep)
                else:
                    hit = (hi[j] + spread >= ep) if buy else (lo[j] <= ep)
                if hit:
                    fill = j
                    break
            if fill is None:
                unfilled += 1
                i += PENDING_BARS  # order kedaluwarsa dulu sebelum sinyal berikutnya dipakai
                continue
            sl_now = s["stop_loss"]
            if (buy and lo[fill] <= sl_now) or ((not buy) and hi[fill] + spread >= sl_now):
                res = ("SL", fill, -1.0)  # terisi dan kena SL di candle yang sama: anggap rugi
            else:
                res = simulate_trade(buy, ep, sl_now, s["take_profit"], spread, hi, lo, cl, fill, max_hold)
            start_i = fill
        else:
            res = simulate_trade(buy, s["entry"], s["stop_loss"], s["take_profit"], spread, hi, lo, cl, i, max_hold)
        if res is None:
            break
        hasil, j, r = res
        trades.append({"side": s["arah"], "i": start_i, "exit": j, "hasil": hasil, "r": r, "skor": s["skor"], "tipe": tipe})
        i = j
    out = summarize_trades(trades)
    out["trades"] = trades
    out["unfilled"] = unfilled
    out["by_tipe"] = {k: sum(1 for t in trades if t["tipe"] == k) for k in ("MARKET", "LIMIT", "STOP")}
    out["start"] = str(df["datetime"].iloc[start])[:10] if "datetime" in df.columns else ""
    out["end"] = str(df["datetime"].iloc[-1])[:10] if "datetime" in df.columns else ""
    out["bars"] = n - start
    return out


def backtest_sync(pair: str, tf: str, n_bars: int) -> dict:
    sym = resolve_symbol(pair)
    info = mt5.symbol_info(sym)
    tick = mt5.symbol_info_tick(sym)
    digits = int(info.digits) if info is not None else 5
    spread = float(tick.ask - tick.bid) if tick is not None else 0.0
    warm = 300
    df = rates_df(sym, TF_MAP[tf], n_bars + warm, warm + 50)
    htf_name, htf_const = HTF_MAP[tf]
    need_h = int((n_bars + warm) * TF_MIN[tf] / HTF_MIN[tf]) + 320
    hdf = rates_df(sym, htf_const, need_h, 100)
    out = run_backtest(df, hdf, htf_name, tf, spread, digits, BT_MAX_HOLD)
    out.update({"spread": spread, "htf_name": htf_name})
    return out


def format_backtest(pair: str, tf: str, r: dict) -> str:
    be = 100.0 / (1.0 + RR)
    lines = [
        f"📈 Backtest {pair} ({tf}), HTF {r['htf_name']}",
        f"Periode: {r['start']} s/d {r['end']} ({r['bars']} candle)",
        f"Aturan: skor minimal {MIN_SCORE}, RR 1:{RR:g}, spread tetap {r['spread']:.5g}",
        "",
    ]
    if r["n"] == 0:
        lines += ["Tidak ada sinyal yang lolos pada periode ini.", "Coba timeframe lain atau tambah jumlah candle."]
    else:
        pf = f"{r['pf']:.2f}" if r["pf"] is not None else ("tak terhingga" if r["gross_win"] > 0 else "-")
        lines += [
            f"Total trade: {r['n']} (BUY {r['buy'][0]}, SELL {r['sell'][0]})",
            f"Jenis: market {r['by_tipe']['MARKET']}, limit {r['by_tipe']['LIMIT']}, stop {r['by_tipe']['STOP']} (tak terisi/batal: {r['unfilled']})",
            f"TP: {r['tp']} | SL: {r['sl']} | Habis waktu: {r['timeout']}",
            f"Win rate: {r['win_rate']:.1f}% (titik impas RR 1:{RR:g} = {be:.1f}%)",
            f"Rata-rata: {r['avg_r']:+.2f}R per trade | Total: {r['total_r']:+.1f}R",
            f"Profit factor: {pf}",
            f"Drawdown maks: {r['max_dd']:.1f}R | Rugi beruntun maks: {r['max_streak']}",
        ]
        if r["n"] < 30:
            lines += ["", f"⚠️ Baru {r['n']} trade: sampel terlalu kecil, hasilnya belum bisa dipercaya (idealnya 30+ trade)."]
        elif r["avg_r"] <= 0:
            lines += ["", "⚠️ Rata-rata hasil tidak positif: aturan ini belum terbukti menguntungkan di periode ini."]
    lines += ["", "Catatan: tanpa komisi/swap/slippage; SL dianggap kena lebih dulu bila SL dan TP satu candle. Hasil lalu tidak menjamin masa depan."]
    return "\n".join(lines)


# ---------- Telegram ----------
def allowed(update: Update) -> bool:
    return not ALLOWED or update.effective_user.id in ALLOWED


MENU_TEXT = "📋 Menu\nPilih pair untuk minta sinyal:"
MENU_ONLY = InlineKeyboardMarkup([[InlineKeyboardButton("📋 Menu", callback_data="menu")]])


def pair_keyboard():
    rows = []
    for i in range(0, len(MENU_PAIRS), 3):
        rows.append([InlineKeyboardButton(p, callback_data=f"p:{p}") for p in MENU_PAIRS[i : i + 3]])
    rows.append([InlineKeyboardButton("💼 Akun", callback_data="akun")])
    return InlineKeyboardMarkup(rows)


def tf_keyboard(pair: str):
    rows = []
    for i in range(0, len(MENU_TFS), 3):
        rows.append([InlineKeyboardButton(t, callback_data=f"m:{pair}:{t}") for t in MENU_TFS[i : i + 3]])
    rows.append([InlineKeyboardButton("⬅️ Kembali", callback_data="menu")])
    return InlineKeyboardMarkup(rows)


def format_signal(pair: str, tf: str, s: dict) -> str:
    ikon = {"BUY": "🟢", "SELL": "🔴", "WAIT": "⏸"}[s["arah"]]
    lines = [
        f"📊 {pair} ({tf}) | HTF {s['htf']}: {s['bias']}",
        f"Arah: {s['arah']} {ikon}",
        f"Skor: {s['skor']}/100 ({s['arah_terbaik']}, minimal {s['min_skor']})",
    ]
    if s["arah"] != "WAIT":
        nama = {
            "MARKET": "MARKET (langsung)",
            "LIMIT": "LIMIT (tunggu pullback)",
            "STOP": "STOP (tunggu tembus level)",
        }[s["tipe"]]
        lines += [f"Order: {s['arah']} {nama}"]
        if s["tipe"] != "MARKET":
            lines.append(f"Harga kini: {s['harga_now']}")
        lines += [f"Entry: {s['entry']}", f"SL: {s['stop_loss']}", f"TP: {s['take_profit']} (RR 1:{RR:g})"]
        if s["tipe"] != "MARKET":
            lines.append("Order tertunda, batal otomatis di akhir hari broker.")
    else:
        lines.append("Ditahan: " + "; ".join(s["blokir"]))
    lines.append("")
    for n, p, m in s["faktor"]:
        mark = "✅" if p == m else ("➖" if p > 0 else "❌")
        lines.append(f"{mark} {n} {p}/{m}")
    lines += ["", f"💬 {s['alasan']}", "", "⚠️ Skor = kekuatan konfluensi teknikal, bukan peluang menang. Bukan saran keuangan."]
    return "\n".join(lines)


async def do_signal(target, ctx: ContextTypes.DEFAULT_TYPE, pair: str, tf: str):
    """Analisis lalu ubah pesan `target` menjadi hasil sinyal (dengan tombol)."""
    rows = []
    try:
        s = await analyze(pair, tf)
        text = format_signal(pair, tf, s)
        if EXEC_ENABLED and s.get("arah") in ("BUY", "SELL") and valid_levels(s):
            key = uuid.uuid4().hex[:8]
            pending = ctx.bot_data.setdefault("pending", {})
            if len(pending) > 50:
                pending.clear()
            pending[key] = (pair, s)
            rows.append(
                [
                    InlineKeyboardButton(f"✅ {s['arah']} {s['tipe']} {LOT} lot", callback_data=f"x:{key}"),
                    InlineKeyboardButton("❌ Batal", callback_data=f"c:{key}"),
                ]
            )
    except Exception as e:
        logging.exception("gagal")
        text = f"Gagal: {e}"
    nav = []
    if pair in MENU_PAIRS and tf in MENU_TFS:
        nav.append(InlineKeyboardButton("🔁 Ulangi", callback_data=f"m:{pair}:{tf}"))
        nav.append(InlineKeyboardButton("📈 Backtest", callback_data=f"b:{pair}:{tf}"))
    nav.append(InlineKeyboardButton("📋 Menu", callback_data="menu"))
    rows.append(nav)
    await target.edit_text(text, reply_markup=InlineKeyboardMarkup(rows))


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    mode = f"Eksekusi MT5: AKTIF ({LOT} lot, tombol konfirmasi)" if EXEC_ENABLED else "Eksekusi MT5: nonaktif (isi ALLOWED_USER_IDS)"
    await update.message.reply_text(
        "Bot sinyal forex (MT5).\n\n"
        "/menu - pilih pair dan timeframe lewat tombol\n"
        "/signal EURUSD [tf] - minta sinyal (tf: 1m, 5m, 15m, 30m, 1h, 4h, 1d)\n"
        "/backtest XAUUSD 15m - uji win rate aturan ini pada riwayat broker\n"
        "/akun - info akun MT5\n"
        "/myid - lihat ID Telegram Anda\n\n"
        f"Sinyal hanya keluar bila skor konfluensi minimal {MIN_SCORE}/100, selain itu WAIT.\n"
        f"{mode}\n"
        "Bukan saran keuangan. Selalu pakai manajemen risiko."
    )


async def menu(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    await update.message.reply_text(MENU_TEXT, reply_markup=pair_keyboard())


async def myid(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"ID Anda: {update.effective_user.id}")


async def akun(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    try:
        await update.message.reply_text(await run_sync(account_text_sync), reply_markup=MENU_ONLY)
    except Exception as e:
        await update.message.reply_text(f"Gagal: {e}")


async def run_backtest_message(target, ctx, pair: str, tf: str, bars: int):
    if ctx.bot_data.get("bt_busy"):
        await target.edit_text("⏳ Backtest lain masih berjalan, tunggu sebentar.", reply_markup=MENU_ONLY)
        return
    ctx.bot_data["bt_busy"] = True
    try:
        text = format_backtest(pair, tf, await run_sync(backtest_sync, pair, tf, bars))
    except Exception as e:
        logging.exception("backtest gagal")
        text = f"Gagal: {e}"
    finally:
        ctx.bot_data["bt_busy"] = False
    await target.edit_text(text, reply_markup=MENU_ONLY)


async def backtest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    if not ctx.args:
        await update.message.reply_text("Contoh: /backtest XAUUSD 15m\nTambah jumlah candle (500-5000): /backtest XAUUSD 15m 3000")
        return
    pair = normalize_pair(ctx.args[0])
    tf = ctx.args[1].lower() if len(ctx.args) > 1 else DEFAULT_TF
    tf = TF_ALIAS.get(tf, tf)
    if tf not in TF_MAP:
        await update.message.reply_text(f"Timeframe valid: {', '.join(TF_MAP)}")
        return
    bars = BT_BARS
    if len(ctx.args) > 2 and ctx.args[2].isdigit():
        bars = max(500, min(int(ctx.args[2]), 5000))
    wait = await update.message.reply_text(f"📈 Backtest {pair} {tf} ({bars} candle) berjalan, bisa 1-2 menit...")
    ctx.application.create_task(run_backtest_message(wait, ctx, pair, tf, bars))


async def signal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    if not ctx.args:
        await update.message.reply_text("Contoh: /signal EURUSD 1h\nAtau ketik /menu untuk pilih lewat tombol.")
        return
    pair = normalize_pair(ctx.args[0])
    tf = ctx.args[1].lower() if len(ctx.args) > 1 else DEFAULT_TF
    tf = TF_ALIAS.get(tf, tf)
    if tf not in TF_MAP:
        await update.message.reply_text(f"Timeframe valid: {', '.join(TF_MAP)}")
        return
    wait = await update.message.reply_text(f"Menganalisis {pair} {tf}...")
    await do_signal(wait, ctx, pair, tf)


async def on_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if not allowed(update):
        return
    data = q.data or ""
    msg = q.message

    if data == "menu":
        await msg.edit_text(MENU_TEXT, reply_markup=pair_keyboard())
        return
    if data.startswith("p:"):
        pair = data[2:]
        if pair in MENU_PAIRS:
            await msg.edit_text(f"{pair}\nPilih timeframe:", reply_markup=tf_keyboard(pair))
        return
    if data.startswith("m:"):
        parts = data.split(":")
        if len(parts) == 3 and parts[1] in MENU_PAIRS and parts[2] in TF_MAP:
            await msg.edit_text(f"Menganalisis {parts[1]} {parts[2]}...")
            await do_signal(msg, ctx, parts[1], parts[2])
        return
    if data.startswith("b:"):
        parts = data.split(":")
        if len(parts) == 3 and parts[1] in MENU_PAIRS and parts[2] in TF_MAP:
            await msg.edit_text(f"📈 Backtest {parts[1]} {parts[2]} berjalan, bisa 1-2 menit...")
            ctx.application.create_task(run_backtest_message(msg, ctx, parts[1], parts[2], BT_BARS))
        return
    if data == "akun":
        try:
            text = await run_sync(account_text_sync)
        except Exception as e:
            text = f"Gagal: {e}"
        await msg.edit_text(text, reply_markup=MENU_ONLY)
        return

    # Tombol eksekusi (x) dan batal (c)
    action, _, key = data.partition(":")
    if action not in ("x", "c"):
        return
    item = ctx.bot_data.get("pending", {}).pop(key, None)
    base = msg.text or ""
    if item is None:
        await msg.edit_text(base + "\n\n⌛ Sinyal sudah kedaluwarsa atau sudah diproses.", reply_markup=MENU_ONLY)
        return
    if action == "c":
        await msg.edit_text(base + "\n\n❌ Dibatalkan.", reply_markup=MENU_ONLY)
        return
    await msg.edit_text(base + "\n\n⏳ Mengirim order...")
    try:
        res = await run_sync(execute_order_sync, *item)
        pend = (item[1].get("tipe") or "MARKET") != "MARKET"
        info_txt = f"Order tertunda terpasang. Tiket: {res.order}" if pend else f"Order terkirim. Tiket: {res.order}, harga: {res.price}"
        await msg.edit_text(base + f"\n\n✅ {info_txt}", reply_markup=MENU_ONLY)
    except Exception as e:
        logging.exception("order gagal")
        await msg.edit_text(base + f"\n\n❌ Order gagal: {e}", reply_markup=MENU_ONLY)


async def post_init(app):
    """Daftarkan perintah ke tombol Menu di Telegram."""
    await app.bot.set_my_commands(
        [
            BotCommand("menu", "Pilih pair dan timeframe"),
            BotCommand("signal", "Minta sinyal, contoh: /signal XAUUSD 15m"),
            BotCommand("backtest", "Uji win rate, contoh: /backtest XAUUSD 15m"),
            BotCommand("akun", "Info akun MT5"),
            BotCommand("myid", "Lihat ID Telegram Anda"),
            BotCommand("start", "Bantuan"),
        ]
    )


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("menu", menu))
    app.add_handler(CommandHandler("myid", myid))
    app.add_handler(CommandHandler("akun", akun))
    app.add_handler(CommandHandler("signal", signal))
    app.add_handler(CommandHandler("backtest", backtest))
    app.add_handler(CallbackQueryHandler(on_button))
    app.run_polling()


if __name__ == "__main__":
    main()
