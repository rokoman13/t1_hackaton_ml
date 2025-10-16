from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import pandas as pd
import sqlite3
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
import os
import glob
import json
from datetime import datetime
import logging
import functools
from functools import lru_cache

# ==================== НАСТРОЙКА ЛОГИРОВАНИЯ ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==================== MODELS ====================
class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5  # Количество похожих вопросов

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    category: str
    subcategory: str
    rating: int

# ==================== AI SYSTEM ====================
class SupportAI:
    def __init__(self, api_token: str, base_url: str = "https://llm.t1v.scibox.tech/v1"):
        self.api_token = api_token
        self.base_url = base_url
        self.chat_client = OpenAI(api_key=api_token, base_url=base_url)
        self.embed_client = OpenAI(api_key=api_token, base_url=base_url)
        self.conn = None
        self.categories = []
        self.subcategories = []  # Все подкатегории из базы
        self.db_file = None
        self._category_embeddings_cache = None
        self._subcategory_embeddings_cache = None
    
    def find_database_file(self) -> bool:
        """Поиск файла базы данных (.db) или CSV для создания базы"""
        # Сначала ищем .db файлы
        db_files = glob.glob("*.db")
        if db_files:
            self.db_file = db_files[0]
            logger.info(f"📁 Найдена база данных: {self.db_file}")
            return True
        
        # Если .db не найдены, ищем CSV файлы
        csv_files = glob.glob("*.csv")
        for csv_file in csv_files:
            # Пропускаем файл с фидбеком
            if "feedback" in csv_file.lower():
                continue
            logger.info(f"📁 Найден CSV файл: {csv_file}. Создаем базу данных...")
            if self.create_new_database_from_csv(csv_file):
                return True
        
        logger.warning("📁 База данных не найдена")
        return False
    
    def create_new_database_from_csv(self, csv_path: str) -> bool:
        """
        Создание новой базы данных из CSV файла
        """
        try:
            logger.info(f"🆕 Создание базы данных из CSV: {csv_path}")
            
            # Загружаем данные
            self.knowledge_df = pd.read_csv(csv_path)
            self.knowledge_df = self.knowledge_df.dropna(subset=['Основная категория'])
            # Проверяем необходимые колонки
            required_columns = ['Основная категория', 'Подкатегория', 'Пример вопроса', 'Шаблонный ответ']
            missing_columns = [col for col in required_columns if col not in self.knowledge_df.columns]
            if missing_columns:
                logger.error(f"❌ В CSV отсутствуют обязательные колонки: {missing_columns}")
                return False
            
            # Проверяем что все подкатегории заполнены
            empty_subcategories = self.knowledge_df['Подкатегория'].isna().sum()
            if empty_subcategories > 0:
                logger.error(f"❌ Найдено {empty_subcategories} записей без подкатегории")
                return False
            
            # Чистим от NaN в основных колонках
            self.knowledge_df = self.knowledge_df.dropna(subset=['Основная категория', 'Пример вопроса'])
            
            logger.info(f"✅ CSV загружен. Записей: {len(self.knowledge_df)}")
            
            # Получаем список категорий и подкатегорий из базы
            self.categories = self.knowledge_df['Основная категория'].unique().tolist()
            self.subcategories = self.knowledge_df['Подкатегория'].unique().tolist()
            
            logger.info(f"📊 Категории в базе: {len(self.categories)}")
            logger.info(f"📊 Подкатегории в базе: {len(self.subcategories)}")
            
            # Логируем распределение по категориям и подкатегориям
            category_stats = self.knowledge_df.groupby('Основная категория')['Подкатегория'].unique()
            for category, subcats in category_stats.items():
                logger.info(f"   📂 {category}: {len(subcats)} подкатегорий - {', '.join(subcats)}")
            
            # Создаем имя для новой базы данных
            base_name = os.path.splitext(os.path.basename(csv_path))[0]
            self.db_file = f"{base_name}.db"
            
            # Создаем SQLite базу
            self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
            self.knowledge_df.to_sql('knowledge_base', self.conn, if_exists='replace', index=False)
            
            # Добавляем колонку для эмбеддингов если её нет
            try:
                self.conn.execute('ALTER TABLE knowledge_base ADD COLUMN embedding BLOB')
            except:
                pass  # Колонка уже существует
            
            # Создаем индексы для быстрого поиска
            self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_category ON knowledge_base ("Основная категория", "Подкатегория")
            ''')
            self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_question ON knowledge_base ("Пример вопроса")
            ''')
            
            self.conn.commit()
            logger.info(f"✅ Новая база данных создана: {self.db_file}")
            
            # ПРЕДВАРИТЕЛЬНО вычисляем эмбеддинги для всей базы
            logger.info("🔄 Вычисляем эмбеддинги для базы знаний...")
            self.precompute_embeddings()
            
            # Предзагружаем кэш категорий и подкатегорий
            self._precache_categories()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания базы данных: {e}")
            return False
    
    def precompute_embeddings(self):
        """
        Предварительное вычисление эмбеддингов для всех примеров вопросов в базе
        """
        try:
            cursor = self.conn.execute('SELECT rowid, "Пример вопроса" FROM knowledge_base')
            items = cursor.fetchall()

            total_items = len(items)
            logger.info(f"📝 Обрабатываем {total_items} вопросов...")

            success_count = 0
            skip_count = 0

            for i, (rowid, question) in enumerate(items):
                # Проверяем, есть ли уже эмбеддинг
                cursor_check = self.conn.execute('SELECT embedding FROM knowledge_base WHERE rowid = ?', (rowid,))
                existing_embedding = cursor_check.fetchone()

                if existing_embedding and existing_embedding[0] is not None:
                    skip_count += 1
                    continue

                # Получаем эмбеддинг для вопроса
                embedding = self.get_embeddings(question)
                if embedding:
                    # Сохраняем как массив float64
                    embedding_array = np.array(embedding, dtype=np.float64)
                    embedding_blob = embedding_array.tobytes()
                    self.conn.execute(
                        'UPDATE knowledge_base SET embedding = ? WHERE rowid = ?',
                        (embedding_blob, rowid)
                    )
                    success_count += 1

                # Коммитим каждые 10 записей
                if (i + 1) % 10 == 0:
                    self.conn.commit()
                    logger.info(f"✅ Обработано {i + 1}/{total_items} вопросов (успешно: {success_count}, пропущено: {skip_count})")

                #time.sleep(0.5)  # Небольшая задержка чтобы не перегружать API

            self.conn.commit()
            logger.info(f"🎯 Все эмбеддинги успешно вычислены и сохранены!")
            logger.info(f"📊 Итоги: успешно {success_count}, пропущено {skip_count}, всего {total_items}")

        except Exception as e:
            logger.error(f"❌ Ошибка при вычислении эмбеддингов: {e}")
    
    def setup_database(self) -> bool:
        if self.find_database_file():
            try:
                # Если база уже подключена в create_new_database_from_csv, просто проверяем
                if self.conn is None:
                    self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
                
                cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge_base'")
                if cursor.fetchone() is None:
                    logger.error("❌ В базе нет таблицы knowledge_base")
                    return False
                
                # Получаем все категории если еще не получены
                if not self.categories:
                    cursor = self.conn.execute('SELECT DISTINCT "Основная категория" FROM knowledge_base')
                    self.categories = [row[0] for row in cursor.fetchall() if row[0]]
                
                # Получаем все подкатегории если еще не получены
                if not self.subcategories:
                    cursor = self.conn.execute('SELECT DISTINCT "Подкатегория" FROM knowledge_base')
                    self.subcategories = [row[0] for row in cursor.fetchall() if row[0]]
                
                # Проверяем что есть категории и подкатегории
                if not self.categories:
                    logger.error("❌ В базе нет категорий")
                    return False
                
                if not self.subcategories:
                    logger.error("❌ В базе нет подкатегорий")
                    return False
                
                logger.info(f"✅ База подключена. Категории: {len(self.categories)}, Подкатегории: {len(self.subcategories)}")
                
                # Логируем распределение подкатегорий по категориям
                logger.info("📋 Распределение подкатегорий по категориям:")
                for category in self.categories:
                    cursor = self.conn.execute(
                        'SELECT DISTINCT "Подкатегория" FROM knowledge_base WHERE "Основная категория" = ?',
                        (category,)
                    )
                    subcats_for_category = [row[0] for row in cursor.fetchall() if row[0]]
                    logger.info(f"   📂 {category}: {len(subcats_for_category)} подкатегорий")
                    if subcats_for_category:
                        logger.info(f"      📁 {', '.join(subcats_for_category)}")
                
                # Проверяем количество записей с эмбеддингами
                cursor = self.conn.execute('SELECT COUNT(*) FROM knowledge_base WHERE embedding IS NOT NULL')
                embedded_count = cursor.fetchone()[0]
                cursor = self.conn.execute('SELECT COUNT(*) FROM knowledge_base')
                total_count = cursor.fetchone()[0]
                logger.info(f"📊 Записей в базе: {total_count}, с эмбеддингами: {embedded_count}")
                
                # Предзагружаем кэш категорий и подкатегорий если еще не загружены
                if self._category_embeddings_cache is None:
                    self._precache_categories()
                
                return True
            except Exception as e:
                logger.error(f"❌ Ошибка подключения к БД: {e}")
                return False
        return False
    
    def _precache_categories(self):
        """Предварительное кэширование эмбеддингов категорий и подкатегорий"""
        logger.info("👁🐽👁 Предзагрузка эмбеддингов категорий и подкатегорий...")
        try:
            # Кэшируем эмбеддинги для категорий
            if self.categories:
                category_embeddings = []
                for category in self.categories:
                    embedding = self.get_embeddings(category)
                    if embedding:
                        category_embeddings.append(embedding)
                    else:
                        logger.error(f"❌ Не удалось получить эмбеддинг для категории: {category}")
                        category_embeddings.append([])  # Добавляем пустой список как заглушку

                if len(category_embeddings) == len(self.categories):
                    self._category_embeddings_cache = category_embeddings
                    logger.info(f"✅ Закэшировано {len(self.categories)} категорий")
                else:
                    logger.error("❌ Не удалось получить эмбеддинги для всех категорий")
                    self._category_embeddings_cache = None

            # Кэшируем эмбеддинги для подкатегорий
            if self.subcategories:
                subcategory_embeddings = []
                for subcategory in self.subcategories:
                    embedding = self.get_embeddings(subcategory)
                    if embedding:
                        subcategory_embeddings.append(embedding)
                    else:
                        logger.error(f"❌ Не удалось получить эмбеддинг для подкатегории: {subcategory}")
                        subcategory_embeddings.append([])  # Добавляем пустой список как заглушку

                if len(subcategory_embeddings) == len(self.subcategories):
                    self._subcategory_embeddings_cache = subcategory_embeddings
                    logger.info(f"✅ Закэшировано {len(self.subcategories)} подкатегорий")
                else:
                    logger.error("❌ Не удалось получить эмбеддинги для всех подкатегорий")
                    self._subcategory_embeddings_cache = None

        except Exception as e:
            logger.error(f"❌ Ошибка кэширования категорий: {e}")
            self._category_embeddings_cache = None
            self._subcategory_embeddings_cache = None

        
    
    @lru_cache(maxsize=1000)
    def get_embeddings(self, text: str) -> Optional[List[float]]:
        """Кэшируем эмбеддинги для одинаковых текстов"""
        try:
            response = self.embed_client.embeddings.create(model="bge-m3", input=[text])
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"❌ Ошибка эмбеддингов: {e}")
            return None
    
    def find_top_subcategories(self, entities_text: str, main_category: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Находим топ-N наиболее подходящих подкатегорий для категории по сущностям"""
        try:
            # Получаем ВСЕ подкатегории для выбранной основной категории из базы
            cursor = self.conn.execute(
                'SELECT DISTINCT "Подкатегория" FROM knowledge_base WHERE "Основная категория" = ?',
                (main_category,)
            )
            available_subcategories = [row[0] for row in cursor.fetchall() if row[0]]

            if not available_subcategories:
                return []

            # Если подкатегорий меньше или равно top_k, возвращаем все
            if len(available_subcategories) <= top_k:
                return [{'subcategory': subcat, 'similarity': 1.0} for subcat in available_subcategories]

            # Если кэш подкатегорий не загружен, возвращаем первые top_k
            if not self._subcategory_embeddings_cache:
                return [{'subcategory': subcat, 'similarity': 1.0} for subcat in available_subcategories[:top_k]]

            # Получаем эмбеддинг сущностей
            entities_embedding = self.get_embeddings(entities_text)
            if not entities_embedding:
                return [{'subcategory': subcat, 'similarity': 1.0} for subcat in available_subcategories[:top_k]]

            # Преобразуем в numpy array
            entities_embedding_array = np.array(entities_embedding, dtype=np.float64).reshape(1, -1)

            # Находим схожести для всех подкатегорий
            subcategory_similarities = []
            for subcat in available_subcategories:
                if subcat in self.subcategories:
                    idx = self.subcategories.index(subcat)
                    if idx < len(self._subcategory_embeddings_cache):
                        subcat_embedding = self._subcategory_embeddings_cache[idx]
                        if not subcat_embedding:
                            continue

                        subcat_embedding_array = np.array(subcat_embedding, dtype=np.float64).reshape(1, -1)

                        if entities_embedding_array.shape[1] != subcat_embedding_array.shape[1]:
                            continue

                        similarity = cosine_similarity(entities_embedding_array, subcat_embedding_array)[0][0]
                        subcategory_similarities.append({'subcategory': subcat, 'similarity': similarity})

            # Сортируем по убыванию схожести и берем топ-N
            subcategory_similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return subcategory_similarities[:top_k]

        except Exception as e:
            logger.error(f"❌ Ошибка поиска топ подкатегорий: {e}")
            return []
        
    def _select_best_category_with_llm(self, user_question: str, entities_text: str, top_candidates: List[Dict]) -> Dict[str, Any]:
        """Выбираем лучшую пару категория-подкатегория через LLM"""
        try:
            # Формируем варианты для LLM
            candidates_text = ""
            for i, candidate in enumerate(top_candidates, 1):
                category = candidate['category']
                similarity = candidate['similarity']
                subcategories = ", ".join([f"{sub['subcategory']} ({sub['similarity']:.3f})" for sub in candidate['subcategories']])
                candidates_text += f"{i}. Категория: {category} (схожесть: {similarity:.3f})\n   Подкатегории: {subcategories}\n"

            prompt = f"""
    Вопрос пользователя: "{user_question}"
    Извлеченные сущности: {entities_text}

    Доступные варианты категорий и подкатегорий:
    {candidates_text}

    Выбери наиболее подходящую пару КАТЕГОРИЯ-ПОДКАТЕГОРИЯ для этого вопроса.
    Будь внимателен: в вопросе может быть название продукта, выглядящее как просто набор слов (например: халва, море, на всё про всё и подобное)
    Будь внимателен: скорее всего клиент спрашивает про какой-либо конктретный продукт банка
    Верни ответ в формате JSON:

    {{
        "selected_category": "название категории",
        "selected_subcategory": "название подкатегории", 
        "confidence": 0.95,
    }}
    """

            response = self.chat_client.chat.completions.create(
                model="qwen2.5-72b-h100",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    'category': result['selected_category'],
                    'subcategory': result['selected_subcategory'],
                    'confidence': result['confidence'],
                    'reasoning': "Автоматический выбор по максимальной схожести + прогон через LLM 2 топ-вариантов"
                }

            # Если не удалось распарсить, возвращаем первый вариант
            first_candidate = top_candidates[0]
            return {
                'category': first_candidate['category'],
                'subcategory': first_candidate['subcategories'][0]['subcategory'],
                'confidence': first_candidate['similarity'],
                'reasoning': "Автоматический выбор по максимальной схожести"
            }

        except Exception as e:
            logger.error(f"❌ Ошибка выбора категории через LLM: {e}")
            first_candidate = top_candidates[0]
            return {
                'category': first_candidate['category'],
                'subcategory': first_candidate['subcategories'][0]['subcategory'],
                'confidence': first_candidate['similarity'],
                'reasoning': "Автоматический выбор из-за ошибки LLM (опять qwen отвалился 🤡)"
            }
    
    def _extract_entities_with_llm(self, user_question: str) -> Dict[str, List[str]]:
        """Извлечение сущностей через LLM"""
        try:
            prompt = f"""
            Извлеки сущности из вопроса клиента банка.
            Будь внимателен: в вопросе может быть название продукта, выглядящее как просто набор слов, вот некоторые примеры:
            Карта "море"
            Карта "форсаж"
            карта "комплимент"
            карта "сигнатура"
            карта "платон"
            карта "кстати"
            карта "черепаха"
            карта "на все про все"
            карта "дальше меньше"
            экспресс-кредит "на роднае"
            рублёвые "суперсемь" и тому подобное
            НЕ ИГНОРИРУЙ подобные слова
            ВОПРОС: "{user_question}"
            
            ВЕРНИ ТОЛЬКО JSON:
            {{
                "entities": {{
                    "products": ["список продуктов"],
                    "actions": ["список действий"], 
                    "problems": ["список проблем"],
                    "objects": ["список объектов"]
                }}
            }}
            """
            
            response = self.chat_client.chat.completions.create(
                model="qwen2.5-72b-h100",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400
            )
            
            result_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            
            entities = {"products": [], "actions": [], "problems": [], "objects": []}
            if json_match:
                entities_data = json.loads(json_match.group())
                entities = entities_data.get("entities", entities)
            
            return entities
            
        except Exception as e:
            logger.error(f"❌ Ошибка извлечения сущностей: {e}")
            return {"products": [], "actions": [], "problems": [], "objects": []}

    def extract_entities_and_classify(self, user_question: str) -> Optional[Dict[str, Any]]:
        logger.info("🔍 Этап 1: Извлечение сущностей и классификация")
        start_time = time.time()

        try:
            # Сначала извлекаем сущности
            entities = self._extract_entities_with_llm(user_question)
            logger.info(f"🏷️ Извлеченные сущности: {entities}")

            # Создаем текст для классификации ТОЛЬКО из сущностей
            entities_text = " ".join([
                " ".join(entities.get('products', [])),
                " ".join(entities.get('actions', [])), 
                " ".join(entities.get('problems', [])),
                " ".join(entities.get('objects', []))
            ]).strip()

            if not entities_text:
                logger.error("❌ Не удалось извлечь сущности из вопроса")
                return None

            logger.info(f"🎯 Классификация по сущностям: {entities_text}")

            # Проверяем что кэш категорий загружен
            if not self._category_embeddings_cache or not self.categories:
                logger.error("❌ Кэш категорий не загружен")
                return None

            # Получаем эмбеддинг для сущностей
            entities_embedding = self.get_embeddings(entities_text)
            if not entities_embedding:
                logger.error("❌ Не удалось получить эмбеддинг сущностей")
                return None

            # Преобразуем в numpy array
            entities_embedding_array = np.array(entities_embedding, dtype=np.float64).reshape(1, -1)

            # Находим ТОП-3 наиболее похожих категорий по сущностям
            category_similarities = []
            for i, category in enumerate(self.categories):
                if i >= len(self._category_embeddings_cache):
                    continue

                category_embedding = self._category_embeddings_cache[i]
                if not category_embedding:
                    continue

                category_embedding_array = np.array(category_embedding, dtype=np.float64).reshape(1, -1)

                if entities_embedding_array.shape[1] != category_embedding_array.shape[1]:
                    continue

                similarity = cosine_similarity(entities_embedding_array, category_embedding_array)[0][0]
                category_similarities.append((category, similarity))

            # Сортируем по убыванию схожести и берем топ-3
            category_similarities.sort(key=lambda x: x[1], reverse=True)
            top_categories = category_similarities[:3]

            logger.info(f"🎯 Топ-3 категорий: {[(cat, round(sim, 3)) for cat, sim in top_categories]}")

            # Находим ТОП-3 подкатегорий для каждой из топ-3 категорий
            top_category_subcategories = []
            for category, similarity in top_categories:
                # ЗДЕСЬ ИСПРАВЛЕНИЕ: используем find_top_subcategories вместо find_best_subcategory
                subcategories = self.find_top_subcategories(entities_text, category, top_k=3)
                top_category_subcategories.append({
                    'category': category,
                    'similarity': similarity,
                    'subcategories': subcategories
                })

            # Выбираем лучшую пару категория-подкатегория через LLM
            best_pair = self._select_best_category_with_llm(user_question, entities_text, top_category_subcategories)

            result = {
                "entities": entities,
                "classification": {
                    "main_category": best_pair['category'],
                    "subcategory": best_pair['subcategory'],
                    "confidence": best_pair['confidence'],
                    "reasoning": best_pair['reasoning']
                },
                "top_candidates": top_category_subcategories
            }

            elapsed_time = time.time() - start_time
            logger.info(f"✅ Классификация завершена за {elapsed_time:.2f}с: {best_pair['category']} → {best_pair['subcategory']}")
            result['timing'] = {'classification': round(elapsed_time, 2)}
            return result

        except Exception as e:
            logger.error(f"❌ Ошибка классификации по сущностям: {e}")
            return None
        
    def semantic_search(self, user_question: str, entities_result: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"🔍 Этап 2: Семантический поиск (top_k={top_k})")
        start_time = time.time()

        # Используем ТОЛЬКО сущности для поиска
        entities_text = ""
        if entities_result and 'entities' in entities_result:
            entities = entities_result['entities']
            entities_text = " ".join([
                " ".join(entities.get('products', [])),
                " ".join(entities.get('actions', [])),
                " ".join(entities.get('problems', [])),
                " ".join(entities.get('objects', []))
            ]).strip()

        if not entities_text:
            logger.warning("⚠️ Нет сущностей для поиска, используем оригинальный вопрос")
            entities_text = user_question

        logger.info(f"🔎 Поиск по сущностям: {entities_text}")

        try:
            entities_embedding = self.get_embeddings(entities_text)
            if not entities_embedding:
                return []

            # Преобразуем в numpy array правильной формы
            entities_embedding_array = np.array(entities_embedding, dtype=np.float64).reshape(1, -1)

            cursor = self.conn.execute('''
                SELECT rowid, "Основная категория", "Подкатегория", "Пример вопроса", "Шаблонный ответ", embedding 
                FROM knowledge_base WHERE embedding IS NOT NULL
            ''')
            knowledge_items = cursor.fetchall()
            logger.info(f"📚 Найдено {len(knowledge_items)} вопросов для поиска")

            similarities = []
            for item in knowledge_items:
                rowid, category, subcategory, example_question, template_answer, embedding_blob = item

                # Загружаем эмбеддинг из базы
                example_embedding = np.frombuffer(embedding_blob, dtype=np.float64)

                # Проверяем размерности
                if entities_embedding_array.shape[1] != example_embedding.shape[0]:
                    logger.warning(f"⚠️ Несовпадение размерностей: сущности {entities_embedding_array.shape[1]}, база {example_embedding.shape[0]}")
                    continue

                similarity = cosine_similarity(entities_embedding_array, [example_embedding])[0][0]

                similarities.append({
                    'rowid': rowid,
                    'category': category,
                    'subcategory': subcategory,
                    'example_question': example_question,
                    'template_answer': template_answer,
                    'similarity': similarity
                })

            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            top_results = similarities[:top_k]

            elapsed_time = time.time() - start_time
            logger.info(f"✅ Найдено {len(top_results)} релевантных результатов по сущностям за {elapsed_time:.2f}с")

            return top_results
        except Exception as e:
            logger.error(f"❌ Ошибка поиска по сущностям: {e}")
            return []
    
    def generate_final_response1(self, user_question: str, entities_result: Dict[str, Any], search_results: List[Dict[str, Any]]) -> str:
        logger.info("🔍 Этап 3: Генерация финального ответа")
        start_time = time.time()
        
        if not search_results:
            logger.warning("⚠️ Похожих вопросов не найдено")
            elapsed_time = time.time() - start_time
            return "❌ В базе знаний не найдено похожих вопросов."
        
        context_items = "\n".join([
            f"{i+1}. {item['category']}/{item['subcategory']} (схожесть: {item['similarity']:.3f})\n   В: {item['example_question']}\n   О: {item['template_answer']}"
            for i, item in enumerate(search_results)
        ])
        
        prompt = f"""
ВОПРОС КЛИЕНТА: "{user_question}"

ПОХОЖИЕ ВОПРОСЫ ИЗ БАЗЫ:
{context_items}

СФОРМИРУЙ ОТВЕТ В ФОРМАТЕ:

🎯 КАТЕГОРИЯ: [категория/подкатегория]
💬 ОТВЕТ ДЛЯ КЛИЕНТА: [ясный и полезный ответ (Используй ТОЛЬКО ОТВЕТ из самой близкой пары Вопрос-ответ без изменений. Ничего не придумывай)]
    
        """
        
        try:
            logger.info("🤖 Генерируем финальный ответ...")
            response = self.chat_client.chat.completions.create(
                model="qwen2.5-72b-h100",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5, #0.3
                max_tokens=1000
            )
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Финальный ответ сгенерирован за {elapsed_time:.2f}с")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ Ошибка генерации ответа: {e}")
            best_match = search_results[0]
            elapsed_time = time.time() - start_time
            return f"""
🎯 КАТЕГОРИЯ: {best_match['category']} → {best_match['subcategory']}
💬 ОТВЕТ ДЛЯ КЛИЕНТА:
{best_match['template_answer']}
            """
 
    def generate_final_response(self, user_question: str, entities_result: Dict[str, Any], search_results: List[Dict[str, Any]]) -> str:
        logger.info("🔍 Этап 3: Генерация финального ответа")
        start_time = time.time()

        if not search_results:
            logger.warning("⚠️ Похожих вопросов не найдено")
            elapsed_time = time.time() - start_time
            return "❌ В базе знаний не найдено похожих вопросов."

        # Компактный формат для похожих вопросов
        
        context_items = "\n".join([
            f"{i+1}. {item['category']}/{item['subcategory']} (схожесть: {item['similarity']:.3f})\n   В: {item['example_question']}\n   О: {item['template_answer']}"
            for i, item in enumerate(search_results)
        ])
        
        prompt = f"""
    ВОПРОС КЛИЕНТА: "{user_question}"

    ПОХОЖИЕ ВОПРОСЫ ИЗ БАЗЫ:
    {context_items}

    СФОРМИРУЙ ОТВЕТ В ФОРМАТЕ:

    🎯 КАТЕГОРИЯ: [категория/подкатегория]

    💬 ОТВЕТ ДЛЯ КЛИЕНТА: [ясный и полезный ответ (Используй ТОЛЬКО ОТВЕТ из самой близкой пары Вопрос-ответ без изменений. Ничего не придумывай)]

    """

        try:
            logger.info("🤖 Генерируем финальный ответ...")
            response = self.chat_client.chat.completions.create(
                model="qwen2.5-72b-h100",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5, #0.3
                max_tokens=1000  
            )
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Финальный ответ сгенерирован за {elapsed_time:.2f}с")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ Ошибка генерации ответа: {e}")
            best_match = search_results[0]
            elapsed_time = time.time() - start_time
            return f"""
    🎯 КАТЕГОРИЯ: {best_match['category']} → {best_match['subcategory']}

    💬 ОТВЕТ ДЛЯ КЛИЕНТА:
    {best_match['template_answer']}

    📊 СХОЖЕСТЬ: {best_match['similarity']:.3f}
    """


    def process_customer_query(self, user_question: str, top_k: int = 5) -> Dict[str, Any]:
        logger.info(f"🚀 Начало обработки запроса: '{user_question}' (top_k={top_k})")
        total_start_time = time.time()
        timing_info = {}
        
        if self.conn is None:
            logger.error("❌ База данных не подключена")
            return {'error': 'База данных не подключена'}
        
        # Этап 1: Классификация
        classification_start_time = time.time()
        entities_result = self.extract_entities_and_classify(user_question)
        if not entities_result:
            logger.error("❌ Не удалось классифицировать вопрос")
            return {'error': 'Не удалось классифицировать вопрос: отвалился qwen 🤡'}
        
        timing_info['classification'] = round(time.time() - classification_start_time, 2)
        
        # Этап 2: Поиск
        search_start_time = time.time()
        search_results = self.semantic_search(user_question, entities_result, top_k=top_k)
        timing_info['search'] = round(time.time() - search_start_time, 2)
        
        # Этап 3: Генерация ответа
        generation_start_time = time.time()
        final_response = self.generate_final_response(user_question, entities_result, search_results)
        timing_info['generation'] = round(time.time() - generation_start_time, 2)
        
        # Общее время
        timing_info['total'] = round(time.time() - total_start_time, 2)
        
        logger.info(f"✅ Обработка запроса завершена за {timing_info['total']}с")
        
        return {
            'user_question': user_question,
            'entities_result': entities_result,
            'search_results': search_results,
            'assistant_response': final_response,
            'timing': timing_info
        }

# ==================== FASTAPI APP ====================
app = FastAPI(title="Support AI System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
support_system = None
feedback_df = pd.DataFrame(columns=['timestamp', 'question', 'answer', 'category', 'subcategory', 'rating'])

@app.on_event("startup")
async def startup_event():
    global support_system
    logger.info("🚀 Инициализация Support AI System...")
    
    API_TOKEN = "sk-x1F57p5wuevcqg5NfOfb7Q"
    support_system = SupportAI(API_TOKEN)
    
    if not support_system.setup_database():
        logger.error("❌ Не удалось подключиться к базе данных")
    else:
        logger.info("✅ Система поддержки инициализирована")
    
    if os.path.exists("user_feedback.csv"):
        global feedback_df
        feedback_df = pd.read_csv("user_feedback.csv")
        logger.info(f"✅ Загружено {len(feedback_df)} фидбеков")

@app.get("/")
async def root():
    logger.info("📊 Получен запрос к корневому эндпоинту")
    return {"message": "Support AI System API", "status": "running"}

@app.post("/process")
async def process_question(request: QuestionRequest):
    logger.info(f"🔧 Получен запрос на обработку: '{request.question}' (top_k={request.top_k})")
    
    if not support_system:
        logger.error("❌ Система не инициализирована")
        raise HTTPException(status_code=500, detail="Система не инициализирована")
    
    start_time = time.time()
    result = support_system.process_customer_query(request.question, request.top_k)
    result['processing_time'] = round(time.time() - start_time, 2)
    
    logger.info(f"⏱️ Общее время обработки: {result['processing_time']} сек")
    return result

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    global feedback_df
    logger.info(f"👍 Получен фидбек для категории: {feedback.category}")
    
    new_feedback = {
        'timestamp': datetime.now().isoformat(),
        'question': feedback.question,
        'answer': feedback.answer,
        'category': feedback.category,
        'subcategory': feedback.subcategory,
        'rating': feedback.rating
    }
    
    feedback_df = pd.concat([feedback_df, pd.DataFrame([new_feedback])], ignore_index=True)
    feedback_df.to_csv("user_feedback.csv", index=False)
    
    logger.info(f"✅ Фидбек сохранен. Всего: {len(feedback_df)}")
    return {"status": "success", "total_feedback": len(feedback_df)}

@app.get("/feedback/stats")
async def get_feedback_stats():
    logger.info("📈 Получен запрос статистики фидбека")
    
    if len(feedback_df) == 0:
        return {"total_feedback": 0, "positive": 0, "negative": 0, "positive_percentage": 0}
    
    total = len(feedback_df)
    positive = len(feedback_df[feedback_df['rating'] == 1])
    negative = len(feedback_df[feedback_df['rating'] == -1])
    
    return {
        "total_feedback": total,
        "positive": positive,
        "negative": negative,
        "positive_percentage": round((positive / total * 100), 1)
    }

if __name__ == "__main__":
    logger.info("🟢 Запуск FastAPI сервера на localhost:8222")
    uvicorn.run(app, host="0.0.0.0", port=8222)