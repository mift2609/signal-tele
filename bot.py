import asyncio
import json
import logging
import os
import uuid

import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

load_dotenv()
logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
AI_API_KEY = os.environ["AI_API_KEY"]
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("AI_MODEL", "qwen-plus")
# WAJIB diisi (ID Telegram dipisah koma) supaya hanya Anda yang bisa memakai bot.
ALLOWED = {int(x) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()}

LOT = float(os.getenv("LOT_SIZE", "0.01"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "3"))
SYMBOL_SUFFIX = os.getenv("SYMBOL_SUFFIX", "")  # contoh "m" kalau broker memakai EURUSDm
ALLOW_REAL = os.getenv("ALLOW_REAL", "0") == "1"  # default: eksekusi hanya di akun DEMO
EXEC_ENABLED = bool(ALLOWED)
MAGIC = 20261020

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
TF_ALIAS = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "60m": "1h", "1d": "1day", "d1": "1day", "h1": "1h", "h4": "4h"}

SYSTEM_PROMPT = (
    "Kamu analis teknikal forex. Berdasarkan data candle dan indikator yang diberikan, "
    "beri sinyal. Balas HANYA JSON valid dengan kunci: "
    '"arah" (BUY/SELL/WAIT), "entry", "stop_loss", "take_profit" (angka), '
    '"keyakinan" (0-100), "alasan" (maks 3 kalimat, Bahasa Indonesia). '
    "Jika kondisi tidak jelas, arah=WAIT dan level null. Rasio risk:reward minimal 1:1.5."
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


def get_candles(pair: str, tf: str) -> pd.DataFrame:
    sym = resolve_symbol(pair)
    rates = mt5.copy_rates_from_pos(sym, TF_MAP[tf], 0, 120)
    if rates is None or len(rates) < 60:
        raise ValueError("Data candle tidak tersedia")
    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(), (df["low"] - df["close"].shift()).abs()],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df


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
    if len(mt5.positions_get() or ()) >= MAX_POSITIONS:
        raise ValueError(f"Batas posisi terbuka ({MAX_POSITIONS}) tercapai")

    sym = resolve_symbol(pair)
    info = mt5.symbol_info(sym)
    tick = mt5.symbol_info_tick(sym)
    if info is None or tick is None:
        raise RuntimeError("Info simbol tidak tersedia")

    buy = s["arah"] == "BUY"
    price = tick.ask if buy else tick.bid
    sl, tp = float(s["stop_loss"]), float(s["take_profit"])
    ok = (sl < price < tp) if buy else (tp < price < sl)
    if not ok:
        raise ValueError(f"SL/TP tidak sesuai harga saat ini ({price}). Minta sinyal baru.")

    fm = info.filling_mode
    if fm & 1:
        filling = mt5.ORDER_FILLING_FOK
    elif fm & 2:
        filling = mt5.ORDER_FILLING_IOC
    else:
        filling = mt5.ORDER_FILLING_RETURN

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
    mode = "DEMO" if a.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else "REAL"
    pos = len(mt5.positions_get() or ())
    return f"Akun {a.login} ({mode})\nServer: {a.server}\nSaldo: {a.balance} {a.currency}\nEkuitas: {a.equity}\nPosisi terbuka: {pos}"


# ---------- AI ----------
async def analyze(pair: str, interval: str) -> dict:
    df = add_indicators(await run_sync(get_candles, pair, interval))
    recent = df.tail(30)[["datetime", "open", "high", "low", "close", "ema20", "ema50", "rsi", "atr"]].round(5)
    msg = f"Pair: {pair}, timeframe: {interval}\nData 30 candle terakhir:\n{recent.to_csv(index=False)}"
    resp = await ai.chat.completions.create(
        model=MODEL,
        max_tokens=600,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": msg}],
    )
    text = resp.choices[0].message.content or ""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("AI tidak mengembalikan JSON, coba kirim ulang perintahnya")
    return json.loads(text[start : end + 1])


# ---------- Telegram ----------
def allowed(update: Update) -> bool:
    return not ALLOWED or update.effective_user.id in ALLOWED


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    mode = f"Eksekusi MT5: AKTIF ({LOT} lot, tombol konfirmasi)" if EXEC_ENABLED else "Eksekusi MT5: nonaktif (isi ALLOWED_USER_IDS)"
    await update.message.reply_text(
        "Bot sinyal forex (MT5).\n\n"
        "/signal EURUSD [tf] - minta sinyal (tf: 1m, 5m, 15m, 30m, 1h, 4h, 1d)\n"
        "/akun - info akun MT5\n"
        "/myid - lihat ID Telegram Anda\n\n"
        f"{mode}\n"
        "Bukan saran keuangan. Selalu pakai manajemen risiko."
    )


async def myid(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"ID Anda: {update.effective_user.id}")


async def akun(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    try:
        await update.message.reply_text(await run_sync(account_text_sync))
    except Exception as e:
        await update.message.reply_text(f"Gagal: {e}")


async def signal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    if not ctx.args:
        await update.message.reply_text("Contoh: /signal EURUSD 1h")
        return
    pair = normalize_pair(ctx.args[0])
    tf = ctx.args[1].lower() if len(ctx.args) > 1 else DEFAULT_TF
    tf = TF_ALIAS.get(tf, tf)
    if tf not in TF_MAP:
        await update.message.reply_text(f"Timeframe valid: {', '.join(TF_MAP)}")
        return
    wait = await update.message.reply_text(f"Menganalisis {pair} {tf}...")
    markup = None
    try:
        s = await analyze(pair, tf)
        text = (
            f"📊 {pair} ({tf})\n"
            f"Arah: {s['arah']}\n"
            f"Entry: {s.get('entry')}\nSL: {s.get('stop_loss')}\nTP: {s.get('take_profit')}\n"
            f"Keyakinan: {s.get('keyakinan')}%\n\n{s['alasan']}\n\n"
            "⚠️ Bukan saran keuangan."
        )
        if EXEC_ENABLED and s.get("arah") in ("BUY", "SELL") and valid_levels(s):
            key = uuid.uuid4().hex[:8]
            pending = ctx.bot_data.setdefault("pending", {})
            if len(pending) > 50:
                pending.clear()
            pending[key] = (pair, s)
            markup = InlineKeyboardMarkup(
                [[
                    InlineKeyboardButton(f"✅ Eksekusi {s['arah']} {LOT} lot", callback_data=f"x:{key}"),
                    InlineKeyboardButton("❌ Batal", callback_data=f"c:{key}"),
                ]]
            )
    except Exception as e:
        logging.exception("gagal")
        text = f"Gagal: {e}"
    await wait.edit_text(text, reply_markup=markup)


async def on_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if not allowed(update):
        return
    action, key = q.data.split(":", 1)
    item = ctx.bot_data.get("pending", {}).pop(key, None)
    base = q.message.text or ""
    if item is None:
        await q.edit_message_text(base + "\n\n⌛ Sinyal sudah kedaluwarsa atau sudah diproses.")
        return
    if action == "c":
        await q.edit_message_text(base + "\n\n❌ Dibatalkan.")
        return
    await q.edit_message_text(base + "\n\n⏳ Mengirim order...")
    try:
        res = await run_sync(execute_order_sync, *item)
        await q.edit_message_text(base + f"\n\n✅ Order terkirim. Tiket: {res.order}, harga: {res.price}")
    except Exception as e:
        logging.exception("order gagal")
        await q.edit_message_text(base + f"\n\n❌ Order gagal: {e}")


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("myid", myid))
    app.add_handler(CommandHandler("akun", akun))
    app.add_handler(CommandHandler("signal", signal))
    app.add_handler(CallbackQueryHandler(on_button))
    app.run_polling()


if __name__ == "__main__":
    main()
