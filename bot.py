"""
Telegram-бот для транслитерации и NLP-распознавания времени.
Команды: /start, /stop, /start_nlp (скрытая), /stop_nlp.
"""

import os
import re
import time
import logging
import requests
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

# NLTK для обработки естественного языка
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

load_dotenv("tokendata.env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BOT_TOKEN = os.environ["BOT_TOKEN"]
API = f"https://api.telegram.org/bot{BOT_TOKEN}"

active_chats = set()        # режим транслитерации
nlp_active_chats = set()    # NLP-режим (время)

# Скачивание данных NLTK при первом запуске
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

RUSSIAN_STOPWORDS = set(stopwords.words('russian'))

# Ключевые слова для распознавания времени
TIME_KEYWORDS = {
    'время', 'time', 'часы', 'сколько', 'который', 'час', 'current time',
    'времени', 'часов', 'минут', 'секунд', 'покажи', 'скажи', 'подскажи',
    'минуты', 'секунды'
}

# Ключевые слова благодарности
THANK_KEYWORDS = {'спасибо', 'благодарю', 'спс', 'отлично', 'хорошо'}

# Ключевые слова для запроса выхода / переключения
EXIT_KEYWORDS = {
    'выход', 'выйти', 'прекрати', 'прекратить', 'закрыть',
    'транслит', 'транслитерация', 'транслитерации',
    'стоп', 'stop', 'exit'
}

# Московский часовой пояс (UTC+3)
MOSCOW_TZ = timezone(timedelta(hours=3))


def detect_time_intent(text: str) -> bool:
    """Проверяет, спрашивает ли пользователь про время."""
    text_lower = text.lower()
    if any(kw in text_lower for kw in TIME_KEYWORDS):
        return True
    try:
        tokens = word_tokenize(text_lower, language='russian')
        filtered = [t for t in tokens if t not in RUSSIAN_STOPWORDS and len(t) > 2]
        for token in filtered:
            if token in TIME_KEYWORDS:
                return True
    except Exception:
        pass
    return False

def detect_thanks(text: str) -> bool:
    """Проверяет, благодарит ли пользователь."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in THANK_KEYWORDS)

def detect_exit_request(text: str) -> bool:
    """Проверяет, спрашивает ли пользователь про выход или переключение."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in EXIT_KEYWORDS)

def get_moscow_time() -> str:
    """Возвращает текущее время по Москве."""
    now = datetime.now(MOSCOW_TZ).strftime("%H:%M:%S")
    return f"Текущее время в Москве (GMT+3): {now}"


# ---------- Транслитерация ----------
BASE_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}
KH_AFTER = {"к", "з", "ц", "с", "е", "х"}

def translit_line(line: str) -> str:
    line = line.lower()
    out = []
    prev = ""
    for ch in line:
        if ch == "х":
            out.append("kh" if prev in KH_AFTER else "h")
        elif ch in BASE_MAP:
            out.append(BASE_MAP[ch])
        elif ch.isspace():
            out.append("_")
        else:
            out.append(ch)
        prev = ch
    result = "".join(out)
    result = re.sub(r"[^a-z0-9_]", "", result)
    result = re.sub(r"_+", "_", result)
    return result.strip("_")

def translit_text(text: str) -> str:
    lines = [l for l in text.splitlines() if l.strip()]
    return "\n".join(translit_line(l) for l in lines)


# ---------- Работа с Telegram API ----------
def tg(method: str, **kwargs) -> dict:
    for attempt in range(3):
        try:
            r = requests.post(f"{API}/{method}", json=kwargs, timeout=30)
            return r.json()
        except requests.RequestException as e:
            if attempt == 2:
                raise
            log.warning("Telegram сеть, попытка %d/3: %s", attempt + 1, e)
            time.sleep(2 ** attempt)

def handle(update: dict) -> None:
    msg = update.get("message") or update.get("channel_post")
    if not msg:
        return
    text = msg.get("text", "").strip()
    if not text:
        return

    chat_id = msg["chat"]["id"]
    reply_to = msg["message_id"]

    # ----- Обработка команд -----
    if text == "/start":
        active_chats.add(chat_id)
        tg("sendMessage", chat_id=chat_id,
           text="Программа активна! Пришли текст, и я транслитерирую его в латиницу.\nДля выхода введите команду /stop",
           reply_to_message_id=reply_to)
        return

    if text == "/stop":
        active_chats.discard(chat_id)
        tg("sendMessage", chat_id=chat_id,
           text="Остановил транслитерацию. Напиши /start, если понадоблюсь снова.",
           reply_to_message_id=reply_to)
        return

    if text == "/start_nlp":
        nlp_active_chats.add(chat_id)
        tg("sendMessage", chat_id=chat_id,
           text="""NLP-режим включён!

Этот модуль разработан в рамках практического задания.
Для выхода напишите /stop_nlp.

Я могу помочь вам узнать текущее время.
Уточните, что вас интересует?""",
           reply_to_message_id=reply_to)
        return

    if text == "/stop_nlp":
        nlp_active_chats.discard(chat_id)
        tg("sendMessage", chat_id=chat_id,
           text="NLP-режим выключен.",
           reply_to_message_id=reply_to)
        return

    if text.startswith("/"):
        return

    # ----- Обычный текст -----

    # Если включён NLP-режим
    if chat_id in nlp_active_chats:
        # Проверяем запрос времени
        if detect_time_intent(text):
            tg("sendMessage", chat_id=chat_id, text=get_moscow_time(), reply_to_message_id=reply_to)
            return

        # Проверяем благодарность
        if detect_thanks(text):
            tg("sendMessage", chat_id=chat_id,
               text="Рад помочь! Если нужно актуализировать данные, спросите ещё раз.",
               reply_to_message_id=reply_to)
            return

        # Проверяем запрос выхода/переключения
        if detect_exit_request(text):
            tg("sendMessage", chat_id=chat_id,
               text="Для выхода из этого режима, введите команду /stop_nlp\nДля возвращения в режим транслита, введите команду /start",
               reply_to_message_id=reply_to)
            return

        # Не распознано
        tg("sendMessage", chat_id=chat_id,
           text="Извините, не понял ваш вопрос. Я могу помочь вам узнать текущее время.\nУточните пожалуйста, что вас интересует?",
           reply_to_message_id=reply_to)
        return

    # Если NLP не включён — проверяем транслитерацию
    if chat_id not in active_chats:
        tg("sendMessage", chat_id=chat_id,
           text="Напиши /start, чтобы перейти в режим транслитерации.",
           reply_to_message_id=reply_to)
        return

    # Транслитерация
    try:
        result = translit_text(text)
        if not result:
            raise ValueError("Не нашёл ни одной буквы для транслитерации")
        tg("sendMessage", chat_id=chat_id, text=result, reply_to_message_id=reply_to)
    except Exception as e:
        log.error("Ошибка транслитерации: %s", e, exc_info=True)
        tg("sendMessage", chat_id=chat_id, text=f" {e}", reply_to_message_id=reply_to)


# ---------- Запуск бота ----------
def get_start_offset() -> int:
    data = tg("getUpdates", offset=-1, timeout=0)
    results = data.get("result", [])
    if results:
        return results[-1]["update_id"] + 1
    return 0

def poll() -> None:
    offset = get_start_offset()
    log.info("Бот запущен, offset=%d", offset)
    while True:
        try:
            data = tg("getUpdates", offset=offset, timeout=30)
            for upd in data.get("result", []):
                offset = upd["update_id"] + 1
                handle(upd)
        except Exception as e:
            log.error("Ошибка polling: %s", e)
            time.sleep(5)

if __name__ == "__main__":
    poll()