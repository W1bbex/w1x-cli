# Contributing to W1x CLI

Спасибо за интерес к проекту W1x CLI! Мы рады любым предложениям по улучшению, пул-реквестам и сообщениям об ошибках.

---

## 🐛 Сообщить об ошибке (Bug Report)

Если вы столкнулись с ошибкой или сбоем:
1. Перейдите в раздел **[GitHub Issues](https://github.com/w1bbex/w1x-cli/issues)**.
2. Нажмите **New Issue** -> выберите **Bug Report**.
3. Укажите:
   - Версию ОС (Windows, Linux, macOS)
   - Версию Python (`python --version`)
   - Модель Ollama, с которой возникла проблема (например: `qwen2.5:7b`, `llama3.2`)
   - Точные шаги воспроизведения и текст ошибки / traceback.

---

## 💡 Предложить идею (Feature Request)

Хотите новый инструмент, команду или улучшение интерфейса?  
Создайте тикет в **[GitHub Issues](https://github.com/w1bbex/w1x-cli/issues)** с тегом `enhancement`.

---

## 🛠 Локальная разработка

1. Склонируйте репозиторий:
   ```bash
   git clone https://github.com/w1bbex/w1x-cli.git
   cd w1x-cli
   ```
2. Установите в режиме разработки:
   ```bash
   pip install -e .
   ```
3. Запустите:
   ```bash
   w1x
   # или напрямую
   python main.py
   ```

---

## 📜 Лицензия

Отправляя свой код, вы соглашаетесь с тем, что ваш вклад будет лицензирован под лицензией **MIT**.
