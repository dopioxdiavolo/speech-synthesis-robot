# Инструкция по загрузке проекта на GitHub

## ✅ Что уже сделано

1. ✅ Git репозиторий инициализирован
2. ✅ Создан `.gitignore` (исключает ненужные файлы)
3. ✅ Все файлы добавлены в git
4. ✅ Создан первый коммит

## 📤 Следующие шаги

### Вариант 1: Через веб-интерфейс GitHub (проще)

1. **Создайте новый репозиторий на GitHub:**
   - Зайдите на https://github.com
   - Нажмите "New repository" (зелёная кнопка)
   - Название: например `speech-synthesis-robot` или `biomimetic-speech-synthesis`
   - Описание: "Система физиологического и просодического моделирования речи для социального робота"
   - Выберите **Public** или **Private**
   - **НЕ** создавайте README, .gitignore или лицензию (они уже есть)
   - Нажмите "Create repository"

2. **Подключите локальный репозиторий к GitHub:**

   Скопируйте URL вашего репозитория (например: `https://github.com/ваш-username/speech-synthesis-robot.git`)

   Затем выполните в терминале:

   ```bash
   cd "/Users/rina/Desktop/ЖОПА ПИЗДА НАХУЙ"
   
   # Переименуйте ветку в main (если нужно)
   git branch -M main
   
   # Добавьте удалённый репозиторий (замените URL на ваш!)
   git remote add origin https://github.com/ваш-username/название-репозитория.git
   
   # Загрузите код на GitHub
   git push -u origin main
   ```

### Вариант 2: Через GitHub CLI (если установлен)

```bash
cd "/Users/rina/Desktop/ЖОПА ПИЗДА НАХУЙ"

# Создать репозиторий и загрузить код одной командой
gh repo create speech-synthesis-robot --public --source=. --remote=origin --push
```

## 🔑 Если потребуется авторизация

Если GitHub попросит логин/пароль:
- Используйте **Personal Access Token** вместо пароля
- Создайте токен: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
- Или используйте GitHub CLI: `gh auth login`

## 📝 Проверка

После загрузки проверьте:
- Откройте ваш репозиторий на GitHub
- Убедитесь, что все файлы загружены
- Проверьте, что README.md отображается корректно

## 🔄 Обновление проекта в будущем

Когда внесёте изменения:

```bash
cd "/Users/rina/Desktop/ЖОПА ПИЗДА НАХУЙ"

# Посмотреть изменения
git status

# Добавить изменения
git add .

# Создать коммит
git commit -m "Описание изменений"

# Загрузить на GitHub
git push
```

## 📦 Что будет загружено

✅ **Будет загружено:**
- Все Python файлы (.py)
- Документация (.md)
- requirements.txt
- .gitignore
- Примеры использования

❌ **НЕ будет загружено** (благодаря .gitignore):
- Аудио файлы (.wav, .mp3)
- Кэш Python (__pycache__)
- Виртуальные окружения (venv/)
- Модели ML (.pth, .pt)
- Временные файлы

---

**Готово!** Ваш проект готов к загрузке на GitHub! 🚀

