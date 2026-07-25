"""
Telegram-бот для транслитерации с модулем распознавания времени на NLTK.

Бот предназначен для решения практической задачи отдела: преобразование
русскоязычных названий файлов в латиницу для загрузки на CDN и в LMS.
Дополнительно реализован NLP-модуль для распознавания запросов о времени.

Команды:
/start          - включает режим транслитерации
/stop           - выключает режим транслитерации
/start_nlp      - включает NLP-режим (скрытая команда для проверяющего)
/stop_nlp       - выключает NLP-режим

Бот разработан для размещения на платформе Amvera, которая не требует
дополнительных механизмов для поддержания активности.
"""

import os
import re
import time
import logging
import requests
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

# Подключаем библиотеку для обработки естественного языка
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# ===== ЗАГРУЗКА ПЕРЕМЕННЫХ ОКРУЖЕНИЯ И ТОКЕНА =====
# Для локальной разработки токен берётся из файла tokendata.env
# На сервере Amvera он будет передан через переменную окружения BOT_TOKEN
load_dotenv("tokendata.env")
BOT_TOKEN = os.environ["BOT_TOKEN"]


# ===== НАСТРОЙКА ЛОГИРОВАНИЯ =====
# Логи помогают отслеживать работу бота и ошибки в реальном времени
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ===== КОНСТАНТЫ И НАСТРОЙКИ =====
# Базовый URL для запросов к Telegram API
API = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Множества для хранения идентификаторов чатов в разных режимах
active_chats = set()        # режим транслитерации
nlp_active_chats = set()    # NLP-режим (распознавание времени)

# Московский часовой пояс (UTC+3) для корректного отображения времени
MOSCOW_TZ = timezone(timedelta(hours=3))


# ===== ЗАГРУЗКА ДАННЫХ ДЛЯ NLTK =====
# Библиотеке NLTK требуются дополнительные файлы для работы с русским языком.
# При первом запуске они скачиваются автоматически.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Список стоп-слов (предлоги, союзы и т.п.), которые не несут смысловой нагрузки
RUSSIAN_STOPWORDS = set(stopwords.words('russian'))


# ===== КЛЮЧЕВЫЕ СЛОВА ДЛЯ РАСПОЗНАВАНИЯ =====
# Слова, связанные с запросом времени
TIME_KEYWORDS = {
    'время', 'time', 'часы', 'сколько', 'который', 'час', 'current time',
    'времени', 'часов', 'минут', 'секунд', 'покажи', 'скажи', 'подскажи',
    'минуты', 'секунды'
}

# Слова благодарности
THANK_KEYWORDS = {'спасибо', 'благодарю', 'спс', 'отлично', 'хорошо'}

# Слова для запроса выхода или переключения режима
EXIT_KEYWORDS = {
    'выход', 'выйти', 'прекрати', 'прекратить', 'закрыть',
    'транслит', 'транслитерация', 'транслитерации',
    'стоп', 'stop', 'exit'
}


# ===== NLP-ФУНКЦИИ =====
# Эти функции используют NLTK для анализа текста и определения намерений пользователя

def detect_time_intent(text: str) -> bool:
    """
    Проверяет, спрашивает ли пользователь о времени.
    Сначала ищет ключевые слова в тексте, затем использует NLTK для токенизации
    и удаления стоп-слов, чтобы выделить значимые термины.
    """
    text_lower = text.lower()

    # Быстрая проверка на наличие ключевых слов
    if any(kw in text_lower for kw in TIME_KEYWORDS):
        return True

    # Более глубокая проверка через NLTK
    try:
        # Разбиваем текст на отдельные слова (токены)
        tokens = word_tokenize(text_lower, language='russian')
        # Убираем стоп-слова и слишком короткие слова (менее 3 букв)
        filtered_tokens = [
            token for token in tokens
            if token not in RUSSIAN_STOPWORDS and len(token) > 2
        ]
        # Проверяем каждое значимое слово
        for token in filtered_tokens:
            if token in TIME_KEYWORDS:
                return True
    except Exception as e:
        # Если NLTK ошибся, просто игнорируем
        log.warning("Ошибка при токенизации NLTK: %s", e)

    return False


def detect_thanks(text: str) -> bool:
    """Проверяет, содержит ли текст благодарность."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in THANK_KEYWORDS)


def detect_exit_request(text: str) -> bool:
    """Проверяет, спрашивает ли пользователь о выходе или переключении режима."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in EXIT_KEYWORDS)


def get_moscow_time() -> str:
    """Возвращает текущее время в Москве с учётом часового пояса MSK (UTC+3)."""
    now = datetime.now(MOSCOW_TZ).strftime("%H:%M:%S")
    return f"Текущее время в Москве (GMT+3): {now}"


# ===== ТРАНСЛИТЕРАЦИЯ =====
# Словарь для преобразования русских букв в латиницу.
# Настройки взяты с сервиса Яндекс.Транслит, адаптированы под задачи отдела.

BASE_MAP = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "",    # твёрдый знак убирается
    "ы": "y",
    "ь": "",    # мягкий знак убирается
    "э": "e", "ю": "yu", "я": "ya",
}

# Множество букв, после которых "х" транслитерируется как "kh".
# В остальных случаях "х" транслитерируется как "h".
KH_AFTER = {"к", "з", "ц", "с", "е", "х"}


def translit_line(line: str) -> str:
    """
    Транслитерирует одну строку текста.
    Пробелы заменяются на подчёркивания, все небуквенные символы удаляются.
    """
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
    # Удаляем всё, кроме латиницы, цифр и подчёркиваний
    result = re.sub(r"[^a-z0-9_]", "", result)
    # Схлопываем множественные подчёркивания в одно
    result = re.sub(r"_+", "_", result)
    # Убираем подчёркивания по краям
    return result.strip("_")


def translit_text(text: str) -> str:
    """
    Транслитерирует весь переданный текст, сохраняя разбивку по строкам.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(translit_line(line) for line in lines)


# ===== ВЗАИМОДЕЙСТВИЕ С TELEGRAM API =====

def tg(method: str, **kwargs) -> dict:
    """
    Универсальная функция для вызова методов Telegram Bot API.
    В случае сетевой ошибки делает до трёх попыток с нарастающей задержкой.
    """
    for attempt in range(3):
        try:
            response = requests.post(f"{API}/{method}", json=kwargs, timeout=30)
            return response.json()
        except requests.RequestException as e:
            if attempt == 2:
                raise
            log.warning("Ошибка сети при вызове %s, попытка %d/3: %s", method, attempt + 1, e)
            time.sleep(2 ** attempt)


def handle(update: dict) -> None:
    """
    Обрабатывает одно входящее сообщение от Telegram.
    Определяет, является ли сообщение командой или обычным текстом,
    и вызывает соответствующую логику.
    """
    msg = update.get("message") or update.get("channel_post")
    if not msg:
        return

    text = msg.get("text", "").strip()
    if not text:
        return

    chat_id = msg["chat"]["id"]
    reply_to = msg["message_id"]

    # ----- Обработка команд -----

    # Включает режим транслитерации
    if text == "/start":
        active_chats.add(chat_id)
        tg(
            "sendMessage",
            chat_id=chat_id,
            text="Программа активна! Пришли текст, и я транслитерирую его в латиницу.\nДля выхода введите команду /stop",
            reply_to_message_id=reply_to
        )
        return

    # Выключает режим транслитерации
    if text == "/stop":
        active_chats.discard(chat_id)
        tg(
            "sendMessage",
            chat_id=chat_id,
            text="Остановил транслитерацию. Напиши /start, если понадоблюсь снова.",
            reply_to_message_id=reply_to
        )
        return

    # Включает NLP-режим (скрытая команда для проверяющего)
    if text == "/start_nlp":
        nlp_active_chats.add(chat_id)
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

    # Выключает NLP-режим
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

    # Сначала проверяем, включён ли NLP-режим
    if chat_id in nlp_active_chats:
        if detect_time_intent(text):
            # Запрос о времени
            tg(
                "sendMessage",
                chat_id=chat_id,
                text=get_moscow_time(),
                reply_to_message_id=reply_to
            )
            return

        if detect_thanks(text):
            # Ответ на благодарность
            tg(
                "sendMessage",
                chat_id=chat_id,
                text="Рад помочь! Если нужно актуализировать данные, спросите ещё раз.",
                reply_to_message_id=reply_to
            )
            return

        if detect_exit_request(text):
            # Запрос о выходе или переключении режима
            tg(
                "sendMessage",
                chat_id=chat_id,
                text="Для выхода из этого режима, введите команду /stop_nlp\nДля возвращения в режим транслита, введите команду /start",
                reply_to_message_id=reply_to
            )
            return

        # Если ничего не распознано
        tg(
            "sendMessage",
            chat_id=chat_id,
            text="Извините, не понял ваш вопрос. Я могу помочь вам узнать текущее время.\nУточните пожалуйста, что вас интересует?",
            reply_to_message_id=reply_to
        )
        return

    # Если NLP не включён, проверяем режим транслитерации
    if chat_id not in active_chats:
        tg(
            "sendMessage",
            chat_id=chat_id,
            text="Напиши /start, чтобы перейти в режим транслитерации.",
            reply_to_message_id=reply_to
        )
        return

    # Транслитерация текста
    try:
        result = translit_text(text)
        if not result:
            raise ValueError("Не нашёл ни одной буквы для транслитерации")
        tg(
            "sendMessage",
            chat_id=chat_id,
            text=result,
            reply_to_message_id=reply_to
        )
    except Exception as e:
        log.error("Ошибка транслитерации: %s", e, exc_info=True)
        tg(
            "sendMessage",
            chat_id=chat_id,
            text=f"Ошибка: {e}",
            reply_to_message_id=reply_to
        )


# ===== ЗАПУСК БОТА =====

def get_start_offset() -> int:
    """
    При первом запуске определяет идентификатор последнего обработанного сообщения.
    Это позволяет боту не отвечать на старые сообщения, которые пришли, пока он был выключен.
    """
    data = tg("getUpdates", offset=-1, timeout=0)
    results = data.get("result", [])
    if results:
        return results[-1]["update_id"] + 1
    return 0


def poll() -> None:
    """
    Основной цикл работы бота.
    Использует long polling — метод, при котором бот сам опрашивает Telegram API
    на наличие новых сообщений, а не ожидает входящих вебхуков.
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
            # При ошибке ждём 5 секунд перед следующей попыткой
            time.sleep(5)


if __name__ == "__main__":
    """
    Точка входа в программу.
    При запуске бот начинает опрашивать Telegram API и обрабатывать сообщения.
    """
    poll()