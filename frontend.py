import streamlit as st
import requests
import pandas as pd
import time

# Конфигурация страницы
st.set_page_config(
    page_title="Support AI System",
    page_icon="🐗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
    }
    .section-header {
        font-size: 1.4rem;
        color: #2e86ab;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2e86ab;
    }
    .entity-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #2e86ab;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .search-result {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #ff6b6b;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .final-answer {
        background: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        white-space: pre-line;
        line-height: 1.6;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .thumbs-btn {
        font-size: 2rem;
        background: none;
        border: none;
        cursor: pointer;
        margin: 0 0.5rem;
        transition: transform 0.2s;
    }
    .thumbs-btn:hover {
        transform: scale(1.2);
    }
    .timing-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .timing-item {
        display: flex;
        justify-content: space-between;
        margin: 0.3rem 0;
        padding: 0.2rem 0;
    }
    .timing-value {
        font-weight: bold;
        color: #d35400;
    }
</style>
""", unsafe_allow_html=True)

API_BASE = "http://localhost:8222"

def init_session_state():
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'feedback_submitted' not in st.session_state:
        st.session_state.feedback_submitted = False
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 3  # Значение по умолчанию

def process_question(question, top_k):
    try:
        response = requests.post(f"{API_BASE}/process", json={"question": question, "top_k": top_k})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Ошибка API: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Ошибка подключения к серверу: {e}")
        return None

def submit_feedback(question, answer, category, subcategory, rating):
    try:
        feedback_data = {
            "question": question,
            "answer": answer,
            "category": category,
            "subcategory": subcategory,
            "rating": rating
        }
        response = requests.post(f"{API_BASE}/feedback", json=feedback_data)
        return response.status_code == 200
    except:
        return False

def get_feedback_stats():
    try:
        response = requests.get(f"{API_BASE}/feedback/stats")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def main():
    init_session_state()
    
    # Заголовок
    st.markdown('<div class="main-header">🐗 Support AI System</div>', unsafe_allow_html=True)
    
    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Настройки поиска")
        
        # Выбор количества похожих вопросов
        top_k = st.slider(
            "Количество похожих вопросов для поиска:",
            min_value=2,
            max_value=5,
            value=st.session_state.top_k,
            help="Меньше значений - быстрее поиск, больше значений - точнее ответ"
        )
        st.session_state.top_k = top_k
        
        st.markdown("---")
        st.header("📊 Статистика системы")
        
        stats = get_feedback_stats()
        if stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Всего отзывов", stats['total_feedback'])
            with col2:
                st.metric("Удовлетворенность", f"{stats['positive_percentage']}%")
            
            col3, col4 = st.columns(2)
            with col3:
                st.metric("👍 Положительные", stats['positive'])
            with col4:
                st.metric("👎 Отрицательные", stats['negative'])
        else:
            st.info("📊 Статистика загружается...")
        
        st.header("💡 Примеры вопросов")
        examples = [
            "Где можно использовать карту MORE?",
            "Как заблокировать потерянную карту?",
            "Какая комиссия за перевод между счетами?",
            "Не работает мобильное приложение",
            "Как получить выписку по счету?",
            "Какие лимиты на снятие наличных?"
        ]
        
        for example in examples:
            if st.button(example, use_container_width=True):
                st.session_state.question_input = example
                st.rerun()
        
        st.markdown("---")
        st.info("""
        **Новые возможности:**
        - 🚀 Кэширование для ускорения работы
        - 🎯 Точный подбор подкатегорий из базы
        - ⚙️ Настройка количества похожих вопросов
        - ⏱️ Детальное время выполнения этапов
        """)

    # Основная область
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Ввод вопроса
        question = st.text_area(
            "💬 **Введите вопрос клиента:**",
            placeholder="Например: Где можно использовать карту MORE? Как пополнить счет? и т.д.",
            height=100,
            key="question_input"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
        with col_btn1:
            if st.button("🚀 Обработать", use_container_width=True, type="primary"):
                if question.strip():
                    with st.spinner(f"🔍 Анализирую вопрос (ищу {top_k} похожих)..."):
                        result = process_question(question, top_k)
                        if result:
                            st.session_state.processed_data = result
                            st.session_state.feedback_submitted = False
                            st.rerun()
                else:
                    st.warning("📝 Пожалуйста, введите вопрос")
        
        with col_btn2:
            if st.button("🔄 Очистить", use_container_width=True):
                st.session_state.processed_data = None
                st.session_state.feedback_submitted = False
                st.rerun()

    # Отображение результатов
    if st.session_state.processed_data:
        data = st.session_state.processed_data
        
        st.markdown("---")
        
        # Время выполнения этапов
        if data.get('timing'):
            timing = data['timing']
    
            # Компактное отображение в одной строке
            timing_text = f"⏱️ Время: общее {timing.get('total', data.get('processing_time', 0)):.1f}с"
            if 'classification' in timing:
                timing_text += f" | классификация (bge + qwen) {timing['classification']:.1f}с"
            if 'search' in timing:
                timing_text += f" | поиск {timing['search']:.1f}с" 
            if 'generation' in timing:
                timing_text += f" | генерация (qwen) {timing['generation']:.1f}с"

            st.info(timing_text)
        
        
        
            if 'error' in data:
                st.error(f"❌ **Ошибка:** {data['error']}")
                return
        
        # ФИНАЛЬНЫЙ ОТВЕТ - В САМОМ ВЕРХУ!
        if data.get('assistant_response'):
            st.markdown("### 💡 Финальный ответ")
            st.markdown('<div class="final-answer">', unsafe_allow_html=True)
            st.write(data['assistant_response'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Кнопки фидбека сразу под ответом
            if not st.session_state.feedback_submitted:
                st.markdown("### 👍👎 Оцените качество ответа")
                col1, col2, col3 = st.columns([1, 1, 8])
                
                classification = data['entities_result'].get('classification', {})
                
                with col1:
                    if st.button("👍 Хорошо", use_container_width=True, type="secondary"):
                        if submit_feedback(
                            data['user_question'],
                            data['assistant_response'],
                            classification.get('main_category', 'Unknown'),
                            classification.get('subcategory', 'Unknown'),
                            1
                        ):
                            st.session_state.feedback_submitted = True
                            st.success("✅ Спасибо за положительный отзыв!")
                            st.rerun()
                
                with col2:
                    if st.button("👎 Плохо", use_container_width=True, type="secondary"):
                        if submit_feedback(
                            data['user_question'],
                            data['assistant_response'],
                            classification.get('main_category', 'Unknown'),
                            classification.get('subcategory', 'Unknown'),
                            -1
                        ):
                            st.session_state.feedback_submitted = True
                            st.info("📝 Спасибо за отзыв! Будем улучшать систему.")
                            st.rerun()
            else:
                st.success("✅ Спасибо за вашу оценку!")
        
        # Информация о процессе обработки
        st.markdown("### 🔍 Детали обработки")
        
        # Извлеченные сущности и классификация
        if data.get('entities_result'):
            entities = data['entities_result'].get('entities', {})
            classification = data['entities_result'].get('classification', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🏷️ Извлеченные сущности")
                with st.container():
                    st.markdown('<div class="entity-box">', unsafe_allow_html=True)
                    
                    if entities.get('products'):
                        st.write("**💳 Продукты:**")
                        for product in entities['products']:
                            st.write(f" - {product}")
                    
                    if entities.get('actions'):
                        st.write("**⚡ Действия:**")
                        for action in entities['actions']:
                            st.write(f" - {action}")
                    
                    if entities.get('problems'):
                        st.write("**🚨 Проблемы:**")
                        for problem in entities['problems']:
                            st.write(f" - {problem}")
                    
                    if entities.get('objects'):
                        st.write("**🏢 Объекты:**")
                        for obj in entities['objects']:
                            st.write(f" - {obj}")
                    
                    if not any(entities.values()):
                        st.write("ℹ️ Сущности не обнаружены")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### 📊 Классификация")
                with st.container():
                    st.markdown('<div class="entity-box">', unsafe_allow_html=True)
                    st.write(f"**📂 Основная категория:** {classification.get('main_category', 'N/A')}")
                    st.write(f"**📁 Подкатегория:** {classification.get('subcategory', 'N/A')}")
                    st.write(f"**🎯 Уверенность:** {classification.get('confidence', 0):.3f}")
                    st.write(f"**💡 Обоснование:** {classification.get('reasoning', 'N/A')}")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Результаты поиска
        if data.get('search_results'):
            st.markdown(f"#### 📚 Найденные похожие вопросы ({len(data['search_results'])} из {st.session_state.top_k})")
            for i, result in enumerate(data['search_results'], 1):
                with st.container():
                    st.markdown('<div class="search-result">', unsafe_allow_html=True)
                    st.write(f"**{i}. 📈 Схожесть: {result.get('similarity', 0):.3f}**")
                    st.write(f"**🏷️ Категория:** {result.get('category', 'N/A')} → {result.get('subcategory', 'N/A')}")
                    st.write(f"**❓ Вопрос:** {result.get('example_question', 'N/A')}")
                    st.write(f"**💡 Ответ из базы:** {result.get('template_answer', 'N/A')}")
                    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()