#!/bin/bash
# Скрипт для загрузки проекта на GitHub
# ЗАМЕНИТЕ URL на ваш репозиторий!

cd "/Users/rina/Desktop/ЖОПА ПИЗДА НАХУЙ"

# Переименуйте ветку в main (если ещё не сделано)
git branch -M main

# Добавьте удалённый репозиторий
# ⚠️ ЗАМЕНИТЕ URL НА ВАШ РЕПОЗИТОРИЙ!
git remote add origin https://github.com/ВАШ-USERNAME/НАЗВАНИЕ-РЕПОЗИТОРИЯ.git

# Загрузите код
git push -u origin main

echo "✅ Проект загружен на GitHub!"

