import asyncio
import json
import logging
import os
import uuid

import MetaTrader5 as mt5
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
    is_demo = a.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO
    pos = len(mt5.positions_get() or ())
    text = f"Akun {a.login} ({'DEMO' if is_demo else 'REAL'})\nServer: {a.server}\nSaldo: {a.balance} {a.currency}\nEkuitas: {a.equity}\nPosisi terbuka: {pos}"
    if not is_demo and not ALLOW_REAL:
        text += "\n\n🔒 Akun REAL: tombol eksekusi order diblokir (hanya sinyal)."
    return text


# ---------- AI ----------
async def analyze(pair: str, interval: str) -> dict:
    df = add_indicators(await run_sync(get_candles, pair, interval))
    recent = df.tail(30)[["datetime", "open", "high", "low", "close", "ema20", "ema50", "rsi", "atr"]].round(5)
    msg = f"Pair: {pair}, timeframe: {interval}\nData 30 candle terakhir:\n{recent.to_csv(index=False)}"
    for _ in range(3):  # coba ulang kalau balasan AI bukan JSON yang valid
        resp = await ai.chat.completions.create(
            model=MODEL,
            max_tokens=600,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": msg}],
        )
        text = resp.choices[0].message.content or ""
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise ValueError("AI tidak mengembalikan JSON setelah 3 percobaan, coba lagi sebentar lagi")


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
    lines = [f"📊 {pair} ({tf})", f"Arah: {s.get('arah')}"]
    if s.get("arah") in ("BUY", "SELL"):
        lines += [f"Entry: {s.get('entry')}", f"SL: {s.get('stop_loss')}", f"TP: {s.get('take_profit')}"]
    lines += [f"Keyakinan: {s.get('keyakinan')}%", "", str(s.get("alasan", "")), "", "⚠️ Bukan saran keuangan."]
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
                    InlineKeyboardButton(f"✅ Eksekusi {s['arah']} {LOT} lot", callback_data=f"x:{key}"),
                    InlineKeyboardButton("❌ Batal", callback_data=f"c:{key}"),
                ]
            )
    except Exception as e:
        logging.exception("gagal")
        text = f"Gagal: {e}"
    nav = []
    if pair in MENU_PAIRS and tf in MENU_TFS:
        nav.append(InlineKeyboardButton("🔁 Ulangi", callback_data=f"m:{pair}:{tf}"))
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
        "/akun - info akun MT5\n"
        "/myid - lihat ID Telegram Anda\n\n"
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
        await msg.edit_text(base + f"\n\n✅ Order terkirim. Tiket: {res.order}, harga: {res.price}", reply_markup=MENU_ONLY)
    except Exception as e:
        logging.exception("order gagal")
        await msg.edit_text(base + f"\n\n❌ Order gagal: {e}", reply_markup=MENU_ONLY)


async def post_init(app):
    """Daftarkan perintah ke tombol Menu di Telegram."""
    await app.bot.set_my_commands(
        [
            BotCommand("menu", "Pilih pair dan timeframe"),
            BotCommand("signal", "Minta sinyal, contoh: /signal XAUUSD 15m"),
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
    app.add_handler(CallbackQueryHandler(on_button))
    app.run_polling()


if __name__ == "__main__":
    main()
