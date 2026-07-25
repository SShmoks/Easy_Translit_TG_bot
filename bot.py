"""
Telegram-бот для транслитерации с модулем распознавания времени на NLTK.

Бот предназначен для преобразования русскоязычного текста в латиницу
и распознавания запросов о времени с помощью библиотеки NLTK.

Команды:
/start          - включить режим транслитерации
/stop           - выключить режим транслитерации
/start_nlp      - включить NLP-режим (скрытая команда для проверяющего)
/stop_nlp       - выключить NLP-режим
"""

import os
import re
import time
import logging
import requests
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

# Библиотека для обработки естественного языка
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# Загрузка переменных окружения
# В локальной среде токен берётся из файла tokendata.env
# На сервере Amvera токен передаётся через переменную окружения BOT_TOKEN
load_dotenv("tokendata.env")
BOT_TOKEN = os.environ["BOT_TOKEN"]

# Настройка логирования для отслеживания работы бота
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# Базовый URL для запросов к Telegram API
API = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Множества для хранения идентификаторов чатов в разных режимах
# active_chats - чаты, где включена транслитерация
# nlp_active_chats - чаты, где включён NLP-режим
active_chats = set()
nlp_active_chats = set()

# Московский часовой пояс (UTC+3)
MOSCOW_TZ = timezone(timedelta(hours=3))


# Загрузка данных для NLTK при первом запуске
# punkt - модель для разбивки текста на слова (токенизация)
# stopwords - список стоп-слов (предлоги, союзы и т.п.)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Стоп-слова русского языка, которые не несут смысловой нагрузки
RUSSIAN_STOPWORDS = set(stopwords.words('russian'))


# Ключевые слова для распознавания запросов о времени
TIME_KEYWORDS = {
    'время', 'time', 'часы', 'сколько', 'который', 'час', 'current time',
    'времени', 'часов', 'минут', 'секунд', 'покажи', 'скажи', 'подскажи',
    'минуты', 'секунды'
}

# Ключевые слова для распознавания благодарности
THANK_KEYWORDS = {'спасибо', 'благодарю', 'спс', 'отлично', 'хорошо'}

# Ключевые слова для распознавания запросов о выходе или переключении
EXIT_KEYWORDS = {
    'выход', 'выйти', 'прекрати', 'прекратить', 'закрыть',
    'транслит', 'транслитерация', 'транслитерации',
    'стоп', 'stop', 'exit'
}


def detect_time_intent(text: str) -> bool:
    """
    Проверяет, спрашивает ли пользователь о времени.

    Сначала выполняет быстрый поиск ключевых слов в тексте.
    Если не находит, использует NLTK для токенизации и удаления стоп-слов,
    после чего проверяет оставшиеся значимые слова.
    """
    text_lower = text.lower()

    # Быстрая проверка на наличие ключевых слов
    if any(kw in text_lower for kw in TIME_KEYWORDS):
        return True

    # Проверка через NLTK
    try:
        # Разбиваем текст на отдельные слова (токены)
        tokens = word_tokenize(text_lower, language='russian')
        # Убираем стоп-слова и слишком короткие слова (менее 3 букв)
        filtered_tokens = [
            token for token in tokens
            if token not in RUSSIAN_STOPWORDS and len(token) > 2
        ]
        # Проверяем каждое значимое слово на наличие в словаре ключевых слов
        for token in filtered_tokens:
            if token in TIME_KEYWORDS:
                return True
    except Exception as e:
        log.warning("Ошибка при токенизации NLTK: %s", e)

    return False


def detect_thanks(text: str) -> bool:
    """
    Проверяет, содержит ли текст благодарность.
    Возвращает True, если найдено любое слово из списка THANK_KEYWORDS.
    """
    text_lower = text.lower()
    return any(kw in text_lower for kw in THANK_KEYWORDS)


def detect_exit_request(text: str) -> bool:
    """
    Проверяет, спрашивает ли пользователь о выходе из режима
    или о переключении на другой режим.
    """
    text_lower = text.lower()
    return any(kw in text_lower for kw in EXIT_KEYWORDS)


def get_moscow_time() -> str:
    """
    Возвращает текущее время в Москве с учётом часового пояса MSK (UTC+3).
    Формат вывода: ЧЧ:ММ:СС
    """
    now = datetime.now(MOSCOW_TZ).strftime("%H:%M:%S")
    return f"Текущее время в Москве (GMT+3): {now}"


# Словарь для транслитерации русских букв в латиницу
# Настройки взяты с сервиса Яндекс.Транслит
# Твёрдый и мягкий знаки удаляются (заменяются на пустую строку)
BASE_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "",    # твёрдый знак удаляется
    "ы": "y",
    "ь": "",    # мягкий знак удаляется
    "э": "e", "ю": "yu", "я": "ya",
}

# Множество букв, после которых "х" транслитерируется как "kh"
# В остальных случаях "х" транслитерируется как "h"
# Правило: если перед "х" стоит к, з, ц, с, е или другая х, то пишем "kh"
KH_AFTER = {"к", "з", "ц", "с", "е", "х"}


def translit_line(line: str) -> str:
    """
    Транслитерирует одну строку текста.

    Пробелы заменяются на подчёркивания для удобства использования в именах файлов.
    Все символы, кроме латиницы, цифр и подчёркиваний, удаляются.
    Множественные подчёркивания схлопываются в одно.
    """
    # Приводим строку к нижнему регистру (все настройки словаря для строчных букв)
    line = line.lower()
    result_parts = []
    prev_char = ""

    for char in line:
        # Специальное правило для буквы "х"
        if char == "х":
            result_parts.append("kh" if prev_char in KH_AFTER else "h")
        elif char in BASE_MAP:
            result_parts.append(BASE_MAP[char])
        elif char.isspace():
            result_parts.append("_")
        else:
            # Цифры, латиница и другие символы оставляем как есть
            result_parts.append(char)
        prev_char = char

    # Формируем итоговую строку
    result = "".join(result_parts)
    # Оставляем только латиницу (a-z), цифры (0-9) и подчёркивания (_)
    result = re.sub(r"[^a-z0-9_]", "", result)
    # Схлопываем множественные подчёркивания в одно
    result = re.sub(r"_+", "_", result)
    # Убираем подчёркивания по краям строки
    return result.strip("_")


def translit_text(text: str) -> str:
    """
    Транслитерирует весь переданный текст, сохраняя разбивку по строкам.

    Пустые строки игнорируются.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(translit_line(line) for line in lines)


def tg(method: str, **kwargs) -> dict:
    """
    Универсальная функция для вызова методов Telegram Bot API.

    В случае сетевой ошибки делает до трёх попыток с нарастающей задержкой.
    Это защита от временных сбоев сети.
    """
    for attempt in range(3):
        try:
            response = requests.post(f"{API}/{method}", json=kwargs, timeout=60)
            return response.json()
        except requests.RequestException as e:
            if attempt == 2:
                raise
            log.warning("Ошибка сети при вызове %s, попытка %d/3: %s", method, attempt + 1, e)
            time.sleep(5)


def handle(update: dict) -> None:
    """
    Обрабатывает одно входящее сообщение от Telegram.

    Определяет тип сообщения (команда или обычный текст) и вызывает
    соответствующую логику обработки.
    """
    # Извлекаем сообщение из update
    msg = update.get("message") or update.get("channel_post")
    if not msg:
        return

    text = msg.get("text", "").strip()
    if not text:
        return

    chat_id = msg["chat"]["id"]
    reply_to = msg["message_id"]


    # ----- Обработка команд -----
    # Команды обрабатываются в первую очередь, независимо от текущего режима


    # /start - включает транслитерацию и выключает NLP, если он был включён
    if text == "/start":
        active_chats.add(chat_id)
        nlp_active_chats.discard(chat_id)  # выключаем NLP при переходе в транслит
        tg(
            "sendMessage",
            chat_id=chat_id,
            text="Программа активна! Пришли текст, и я транслитерирую его в латиницу.\nДля выхода введите команду /stop",
            reply_to_message_id=reply_to
        )
        return

    # /stop - выключает транслитерацию, но не влияет на NLP
    if text == "/stop":
        active_chats.discard(chat_id)
        tg(
            "sendMessage",
            chat_id=chat_id,
            text="Остановил транслитерацию.",
            reply_to_message_id=reply_to
        )
        return

    # /start_nlp - включает NLP и выключает транслитерацию, если она была включена
    if text == "/start_nlp":
        nlp_active_chats.add(chat_id)
        active_chats.discard(chat_id)  # выключаем транслит при переходе в NLP
        tg(
            "sendMessage",
            chat_id=chat_id,
            text="""NLP-режим включён!

Этот модуль разработан в рамках практического задания.
Для выхода напишите /stop_nlp.

Я могу помочь вам узнать текущее время.
Уточните, что вас интересует?""",
            reply_to_message_id=reply_to
        )
        return

    # /stop_nlp - выключает NLP, но не влияет на транслитерацию
    if text == "/stop_nlp":
        nlp_active_chats.discard(chat_id)
        tg(
            "sendMessage",
            chat_id=chat_id,
            text="NLP-режим выключен.",
            reply_to_message_id=reply_to
        )
        return

    # Любая неизвестная команда игнорируется
    if text.startswith("/"):
        return


    # ----- Обработка обычного текста -----

    # Приоритет у NLP-режима: если он включён, проверяем запросы на время
    if chat_id in nlp_active_chats:
        # Запрос о времени
        if detect_time_intent(text):
            tg("sendMessage", chat_id=chat_id, text=get_moscow_time(), reply_to_message_id=reply_to)
            return

        # Благодарность
        if detect_thanks(text):
            tg("sendMessage", chat_id=chat_id, text="Рад помочь! Если нужно актуализировать данные, спросите ещё раз.", reply_to_message_id=reply_to)
            return

        # Запрос о выходе или переключении
        if detect_exit_request(text):
            tg("sendMessage", chat_id=chat_id, text="Для выхода из этого режима, введите команду /stop_nlp\nДля возвращения в режим транслита, введите команду /start", reply_to_message_id=reply_to)
            return

        # Если ничего не распознано
        tg("sendMessage", chat_id=chat_id, text="Извините, не понял ваш вопрос. Я могу помочь вам узнать текущее время.\nУточните пожалуйста, что вас интересует?", reply_to_message_id=reply_to)
        return

    # Если NLP не включён — проверяем транслитерацию
    if chat_id not in active_chats:
        # Режим ожидания: ни один режим не активен
        tg("sendMessage", chat_id=chat_id, text="Напиши /start, чтобы перейти в режим транслитерации.", reply_to_message_id=reply_to)
        return

    # Режим транслитерации активен — преобразуем текст
    try:
        result = translit_text(text)
        if not result:
            raise ValueError("Не нашёл ни одной буквы для транслитерации")
        tg("sendMessage", chat_id=chat_id, text=result, reply_to_message_id=reply_to)
    except Exception as e:
        log.error("Ошибка транслитерации: %s", e, exc_info=True)
        tg("sendMessage", chat_id=chat_id, text=f"Ошибка: {e}", reply_to_message_id=reply_to)


def get_start_offset() -> int:
    """
    При первом запуске определяет идентификатор последнего обработанного сообщения.

    Это позволяет боту не отвечать на старые сообщения, которые пришли,
    пока он был выключен.
    """
    data = tg("getUpdates", offset=-1, timeout=0)
    results = data.get("result", [])
    if results:
        return results[-1]["update_id"] + 1
    return 0


def poll() -> None:
    """
    Основной цикл работы бота.

    Использует метод long polling: бот сам опрашивает Telegram API
    на наличие новых сообщений, а не ожидает входящих вебхуков.
    Это позволяет работать без публичного URL.
    """
    offset = get_start_offset()
    log.info("Бот запущен, offset=%d", offset)

    while True:
        try:
            # Запрос к Telegram API с таймаутом 30 секунд
            data = tg("getUpdates", offset=offset, timeout=30)

            for update in data.get("result", []):
                offset = update["update_id"] + 1
                handle(update)

        except Exception as e:
            log.error("Ошибка в основном цикле: %s", e)
            time.sleep(5)


if __name__ == "__main__":
    """
    Точка входа в программу.
    При запуске бот начинает опрашивать Telegram API и обрабатывать сообщения.
    """
    poll()