import subprocess
import time
import sys
import os
import logging
import glob

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Проверка установленных зависимостей"""
    try:
        import fastapi, streamlit, openai, pandas, sklearn
        logger.info("✅ Все зависимости установлены")
        return True
    except ImportError as e:
        logger.error(f"❌ Отсутствуют зависимости: {e}")
        logger.info("📦 Установите: pip install -r requirements.txt")
        return False

def find_data_files():
    """Поиск файлов с данными"""
    db_files = glob.glob("*.db") + glob.glob("data/*.db")
    csv_files = [f for f in glob.glob("*.csv") + glob.glob("data/*.csv") if "feedback" not in f.lower()]
    
    return db_files, csv_files

def run_system():
    """Запуск всей системы"""
    logger.info("🚀 Запуск Support AI System в Docker...")
    print("=" * 60)
    
    if not check_dependencies():
        return
    
    # Проверяем наличие файлов данных
    db_files, csv_files = find_data_files()
    
    if db_files:
        logger.info(f"Найдены базы данных: {', '.join(db_files)}")
    if csv_files:
        logger.info(f"Найдены CSV файлы: {', '.join(csv_files)}")
    
    if not db_files and not csv_files:
        logger.error("Не найдены файлы данных (.db или .csv)")
        logger.info("Поместите файл базы знаний (.db) или CSV файл в директорию data/ или текущую директорию")
        logger.info("Монтируйте volume с данными при запуске Docker")
        return
    
    if csv_files and not db_files:
        logger.info("🆕 Будет создана новая база данных из CSV файла")
    
    # Запускаем бекенд в фоновом режиме
    logger.info("🔧 Запуск бекенда FastAPI...")
    backend_process = subprocess.Popen([
        sys.executable, "backend.py"
    ])
    
    # Ждем запуска бекенда (дольше если создается новая база)
    wait_time = 15 if csv_files and not db_files else 8
    logger.info(f"Ожидание запуска бекенда ({wait_time}сек)...")
    time.sleep(wait_time)
    
    # Запускаем фронтенд
    logger.info("🖥️ Запуск фронтенда Streamlit...")
    frontend_process = subprocess.Popen([
        "streamlit", "run", "frontend.py", 
        "--server.port", "8501",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--server.address", "0.0.0.0"
    ])
    
    print("=" * 60)
    logger.info("✅ Система успешно запущена!")
    print("🌐 Фронтенд доступен по адресу: http://localhost:8501")
    print("🔗 API доступно по адресу: http://localhost:8222")
    print("📊 Документация API: http://localhost:8222/docs")
    print("")
    print("🛑 Для остановки нажмите Ctrl+C")
    
    try:
        # Ожидаем завершения процессов
        backend_process.wait()
        frontend_process.wait()
    except KeyboardInterrupt:
        logger.info("Остановка системы...")
        backend_process.terminate()
        frontend_process.terminate()
        logger.info("✅ Система остановлена")

if __name__ == "__main__":
    run_system()