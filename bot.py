import json
import logging
import os

import httpx
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

load_dotenv()
logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
AI_API_KEY = os.environ["AI_API_KEY"]
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
TWELVEDATA_KEY = os.environ["TWELVEDATA_API_KEY"]
MODEL = os.getenv("AI_MODEL", "qwen-plus")
# Kosongkan = semua orang boleh pakai. Isi ID Telegram dipisah koma untuk membatasi.
ALLOWED = {int(x) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()}

ai = AsyncOpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)
DEFAULT_TF = "1h"
VALID_TF = {"15min", "30min", "1h", "4h", "1day"}

SYSTEM_PROMPT = (
    "Kamu analis teknikal forex. Berdasarkan data candle dan indikator yang diberikan, "
    "beri sinyal. Balas HANYA JSON valid dengan kunci: "
    '"arah" (BUY/SELL/WAIT), "entry", "stop_loss", "take_profit" (angka), '
    '"keyakinan" (0-100), "alasan" (maks 3 kalimat, Bahasa Indonesia). '
    "Jika kondisi tidak jelas, arah=WAIT dan level null. Rasio risk:reward minimal 1:1.5."
)


def normalize_pair(raw: str) -> str:
    p = raw.upper().replace("/", "").replace("-", "")
    return f"{p[:3]}/{p[3:]}" if len(p) == 6 else raw.upper()


async def fetch_candles(symbol: str, interval: str) -> pd.DataFrame:
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(
            "https://api.twelvedata.com/time_series",
            params={"symbol": symbol, "interval": interval, "outputsize": 120, "apikey": TWELVEDATA_KEY},
        )
    data = r.json()
    if "values" not in data:
        raise ValueError(data.get("message", "Data tidak tersedia"))
    df = pd.DataFrame(data["values"]).iloc[::-1].reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
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


async def analyze(symbol: str, interval: str) -> dict:
    df = add_indicators(await fetch_candles(symbol, interval))
    recent = df.tail(30)[["datetime", "open", "high", "low", "close", "ema20", "ema50", "rsi", "atr"]].round(5)
    msg = f"Pair: {symbol}, timeframe: {interval}\nData 30 candle terakhir:\n{recent.to_csv(index=False)}"
    resp = await ai.chat.completions.create(
        model=MODEL,
        max_tokens=600,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": msg}],
    )
    text = resp.choices[0].message.content.strip().strip("`")
    if text.lower().startswith("json"):
        text = text[4:]
    return json.loads(text)


def allowed(update: Update) -> bool:
    return not ALLOWED or update.effective_user.id in ALLOWED


async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    await update.message.reply_text(
        "Bot sinyal forex.\n\n"
        "/signal EURUSD [tf] - minta sinyal (tf: 15min, 30min, 1h, 4h, 1day)\n"
        "/myid - lihat ID Telegram Anda\n\n"
        "Bukan saran keuangan. Selalu pakai manajemen risiko."
    )


async def myid(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"ID Anda: {update.effective_user.id}")


async def signal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not allowed(update):
        return
    if not ctx.args:
        await update.message.reply_text("Contoh: /signal EURUSD 1h")
        return
    symbol = normalize_pair(ctx.args[0])
    tf = ctx.args[1].lower() if len(ctx.args) > 1 else DEFAULT_TF
    if tf not in VALID_TF:
        await update.message.reply_text(f"Timeframe valid: {', '.join(sorted(VALID_TF))}")
        return
    wait = await update.message.reply_text(f"Menganalisis {symbol} {tf}...")
    try:
        s = await analyze(symbol, tf)
        text = (
            f"📊 {symbol} ({tf})\n"
            f"Arah: {s['arah']}\n"
            f"Entry: {s.get('entry')}\nSL: {s.get('stop_loss')}\nTP: {s.get('take_profit')}\n"
            f"Keyakinan: {s.get('keyakinan')}%\n\n{s['alasan']}\n\n"
            "⚠️ Bukan saran keuangan."
        )
    except Exception as e:
        logging.exception("gagal")
        text = f"Gagal: {e}"
    await wait.edit_text(text)


def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("myid", myid))
    app.add_handler(CommandHandler("signal", signal))
    app.run_polling()


if __name__ == "__main__":
    main()
