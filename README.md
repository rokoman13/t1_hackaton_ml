# Support AI System 🐗

Система искусственного интеллекта для поддержки клиентов банка с использованием FastAPI и Streamlit.

## 🚀 Быстрый запуск

### Способ 1: Docker Compose (рекомендуется)

1. Убедитесь, что у вас установлены Docker и Docker Compose
2. Поместите ваш CSV файл с базой знаний в папку `data/` (создайте если нет)
3. Выполните команды:

```bash
# Создайте папку для данных
mkdir -p data

# Поместите ваш CSV файл в папку data/
сp smart_support_vtb_belarus_faq_final-Copy1.csv data/

# Запустите приложение
docker build -t support-ai .

docker run -it -p 8222:8222 -p 8501:8501 -v $(pwd)/data:/app/data support-ai
