FROM python:3.11-slim

WORKDIR /app

# Копируем requirements первым для кэширования слоев
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта
COPY . .

# Создаем volume для хранения данных
VOLUME /app/data

# Открываем порты для backend и frontend
EXPOSE 8222 8501

# Запускаем приложение
CMD ["python", "run.py"]