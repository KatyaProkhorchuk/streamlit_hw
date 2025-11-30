from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from task3 import draw_plot
from task5 import build_map
from task6 import load_price_model
from task7 import get_yandex_gpt_openai_response
import os

if not os.path.exists("1/computer_prices_all.csv"):
    import kagglehub
    import shutil
    # Download latest version
    path = kagglehub.dataset_download("paperxd/all-computer-prices")

    print("Path to dataset files:", path)
    shutil.move(path, "1/")

st.set_page_config(
    page_title="Computer price analise",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_data():
    try:
        data = pd.read_csv("1/computer_prices_all.csv")
        return data
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return pd.DataFrame()

df = load_data()

st.sidebar.title("💻 Анализ цен на компьютеры")
st.sidebar.write("---")
page = st.sidebar.radio("Навигация", [
    "Описание проекта",
    "Статистика",
    "Динамика выпуска",
    "Вопрос-ответ",
    "Производители",
    "Предсказание цен",
    "Чат с ИИ"
])

# =========================================
# 1: ОПИСАНИЕ ПРОЕКТА
# =========================================
if page == "Описание проекта":
    st.title("Анализ рынка компьютерной техники")

    st.markdown("""
    Это интерактивное веб-приложение для анализа датасета цен на компьютерную технику. 
    Приложение позволяет исследовать различные аспекты рынка компьютеров, включая:

    - Статистический анализ
    - Временные тренды
    - Карта штаб квартир
    - Работа с моделью
    - ИИ-помощник
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.image("https://img.freepik.com/premium-photo/business-analyst-financial-adviser-cat-working-generative-ai_842983-381.jpg",
                 caption="Рисунок 1: Котик анализирует рынок компьютероа")

    with col2:
        st.image("https://www.itbestsellers.ru/etc/Logo/532652.png",
                 caption="Рисунок 2: Тренды цен 2024 в РФ")

    st.markdown("""
    ### Пример данных в датасете
    """)

    st.dataframe(df.head(), use_container_width=True)

    st.markdown("""
    ### Технологии
    - Streamlit 
    - Pandas 
    - Plotly 
    - Scikit-learn
    """)

# =========================================
# 2: ОБЩАЯ СТАТИСТИКА
# =========================================
elif page == "Статистика":
    st.title("📊 Общая статистика датасета")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Основные характеристики", "Процессоры и память", "Графика и дисплеи", "Цены и гарантии"])
    with tab1:
        st.subheader("Основные характеристики устройств")
        col1, col2 = st.columns(2)

        with col1:
            device_counts = df['device_type'].value_counts()
            fig = px.pie(
                device_counts,
                values=device_counts.values,
                names=device_counts.index,
                title='Распределение по типу устройств',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            brand_counts = df['brand'].value_counts().head(10)
            fig = px.bar(
                brand_counts,
                x=brand_counts.values,
                y=brand_counts.index,
                orientation='h',
                title='Топ-10 производителей',
                labels={'x': 'Количество устройств', 'y': 'Бренд'},
                color=brand_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            os_counts = df['os'].value_counts().head(8)
            fig = px.pie(
                os_counts,
                values=os_counts.values,
                names=os_counts.index,
                title='Распределение операционных систем',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            form_factor_counts = df['form_factor'].value_counts()
            fig = px.bar(
                form_factor_counts,
                x=form_factor_counts.index,
                y=form_factor_counts.values,
                title='Распределение по форм-факторам',
                labels={'x': 'Форм-фактор', 'y': 'Количество'},
                color=form_factor_counts.values,
                color_continuous_scale='Viridis'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Процессоры и память")
        col1, col2, col3 = st.columns(3)

        with col1:
            cpu_brand_counts = df['cpu_brand'].value_counts()
            fig = px.pie(
                cpu_brand_counts,
                values=cpu_brand_counts.values,
                names=cpu_brand_counts.index,
                title='Бренды процессоров'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            cpu_tier_counts = df['cpu_tier'].value_counts()
            fig = px.bar(
                cpu_tier_counts,
                x=cpu_tier_counts.index,
                y=cpu_tier_counts.values,
                title='Уровни процессоров',
                color=cpu_tier_counts.values
            )
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            cpu_cores_counts = df['cpu_cores'].value_counts().sort_index()
            fig = px.bar(
                cpu_cores_counts,
                x=cpu_cores_counts.index,
                y=cpu_cores_counts.values,
                title='Распределение по количеству ядер',
                labels={'x': 'Количество ядер', 'y': 'Устройств'}
            )
            st.plotly_chart(fig, use_container_width=True)

        col4, col5 = st.columns(2)

        with col4:
            ram_counts = df['ram_gb'].value_counts().sort_index()
            fig = px.bar(
                ram_counts,
                x=ram_counts.index,
                y=ram_counts.values,
                title='Объем оперативной памяти (ГБ)',
                color=ram_counts.values,
                color_continuous_scale='Teal'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col5:
            storage_type_counts = df['storage_type'].value_counts()
            fig = px.pie(
                storage_type_counts,
                values=storage_type_counts.values,
                names=storage_type_counts.index,
                title='Типы накопителей'
            )
            st.plotly_chart(fig, use_container_width=True)

        col6, col7 = st.columns(2)

        with col6:
            if 'storage_gb' in df.columns:
                storage_bins = [0, 256, 512, 1024, 2048, float('inf')]
                # увы нет поддержки латеха, пришлось вставлять юникод символы
                storage_labels = ['≤256GB', '257-512GB', '513GB-1TB', '1-2TB', '>2TB']

                df['storage_group'] = pd.cut(df['storage_gb'], bins=storage_bins, labels=storage_labels)
                storage_group_counts = df['storage_group'].value_counts()

                fig = px.bar(
                    storage_group_counts,
                    x=storage_group_counts.index,
                    y=storage_group_counts.values,
                    title='Группы объемов хранилища',
                    color=storage_group_counts.values
                )
                st.plotly_chart(fig, use_container_width=True)

        with col7:
            if 'storage_drive_count' in df.columns:
                drive_counts = df['storage_drive_count'].value_counts().sort_index()
                fig = px.pie(
                    drive_counts,
                    values=drive_counts.values,
                    names=drive_counts.index,
                    title='Количество накопителей'
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Графика и дисплеи")
        col1, col2, col3 = st.columns(3)

        with col1:
            gpu_brand_counts = df['gpu_brand'].value_counts()
            fig = px.pie(
                gpu_brand_counts,
                values=gpu_brand_counts.values,
                names=gpu_brand_counts.index,
                title='Бренды видеокарт'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            gpu_tier_counts = df['gpu_tier'].value_counts()
            fig = px.bar(
                gpu_tier_counts,
                x=gpu_tier_counts.index,
                y=gpu_tier_counts.values,
                title='Уровни видеокарт',
                color=gpu_tier_counts.values
            )
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            if 'vram_gb' in df.columns:
                vram_counts = df['vram_gb'].value_counts().sort_index()
                fig = px.bar(
                    vram_counts,
                    x=vram_counts.index,
                    y=vram_counts.values,
                    title='Объем видеопамяти (ГБ)',
                    color=vram_counts.values
                )
                st.plotly_chart(fig, use_container_width=True)

        col4, col5 = st.columns(2)

        with col4:
            if 'display_type' in df.columns:
                display_type_counts = df['display_type'].value_counts().head(8)
                fig = px.pie(
                    display_type_counts,
                    values=display_type_counts.values,
                    names=display_type_counts.index,
                    title='Типы дисплеев'
                )
                st.plotly_chart(fig, use_container_width=True)

        with col5:
            if 'display_size_in' in df.columns:
                display_bins = [0, 13, 15, 17, 20, float('inf')]
                display_labels = ['≤13"', '14-15"', '16-17"', '18-20"', '>20"']
                df['display_group'] = pd.cut(df['display_size_in'], bins=display_bins, labels=display_labels)
                display_group_counts = df['display_group'].value_counts()

                fig = px.bar(
                    display_group_counts,
                    x=display_group_counts.index,
                    y=display_group_counts.values,
                    title='Размеры дисплеев',
                    color=display_group_counts.values
                )
                st.plotly_chart(fig, use_container_width=True)

        col6, col7 = st.columns(2)

        with col6:
            if 'resolution' in df.columns:
                resolution_counts = df['resolution'].value_counts().head(10)
                fig = px.bar(
                    resolution_counts,
                    x=resolution_counts.values,
                    y=resolution_counts.index,
                    orientation='h',
                    title='Топ-10 разрешений экранов',
                    color=resolution_counts.values
                )
                st.plotly_chart(fig, use_container_width=True)

        with col7:
            if 'refresh_hz' in df.columns:
                refresh_counts = df['refresh_hz'].value_counts().sort_index().head(15)
                fig = px.bar(
                    refresh_counts,
                    x=refresh_counts.index,
                    y=refresh_counts.values,
                    title='Частота обновления (Гц)',
                    color=refresh_counts.values
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Цены, батареи и гарантии")
        col1, col2 = st.columns(2)

        with col1:
            fig = px.histogram(
                df,
                x='price',
                nbins=50,
                title='Распределение цен',
                labels={'price': 'Цена ($)'},
                color_discrete_sequence=['#FF6B6B']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            top_brands = df['brand'].value_counts().head(10).index
            df_top_brands = df[df['brand'].isin(top_brands)]

            fig = px.box(
                df_top_brands,
                x='brand',
                y='price',
                title='Распределение цен по топ-брендам',
                labels={'brand': 'Бренд', 'price': 'Цена ($)'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4, col5 = st.columns(3)
        with col3:
            if 'battery_wh' in df.columns:
                fig = px.histogram(
                    df,
                    x='battery_wh',
                    nbins=30,
                    title='Емкость батареи (Wh)',
                    color_discrete_sequence=['#4ECDC4']
                )
                st.plotly_chart(fig, use_container_width=True)

        with col4:
            if 'charger_watts' in df.columns:
                fig = px.histogram(
                    df,
                    x='charger_watts',
                    nbins=30,
                    title='Мощность зарядки (Вт)',
                    color_discrete_sequence=['#45B7D1']
                )
                st.plotly_chart(fig, use_container_width=True)

        with col5:
            if 'psu_watts' in df.columns:
                fig = px.histogram(
                    df,
                    x='psu_watts',
                    nbins=30,
                    title='Блоки питания (Вт)',
                    color_discrete_sequence=['#96CEB4']
                )
                st.plotly_chart(fig, use_container_width=True)

        col6, col7 = st.columns(2)

        with col6:
            if 'weight_kg' in df.columns:
                fig = px.histogram(
                    df,
                    x='weight_kg',
                    nbins=30,
                    title='Вес устройств (кг)',
                    color_discrete_sequence=['#FECA57']
                )
                st.plotly_chart(fig, use_container_width=True)

        with col7:
            if 'warranty_months' in df.columns:
                warranty_counts = df['warranty_months'].value_counts().sort_index()
                fig = px.bar(
                    warranty_counts,
                    x=warranty_counts.index,
                    y=warranty_counts.values,
                    title='Срок гарантии (месяцев)',
                    color=warranty_counts.values
                )
                st.plotly_chart(fig, use_container_width=True)

        col8, col9 = st.columns(2)

        with col8:
            if 'wifi' in df.columns:
                wifi_counts = df['wifi'].value_counts()
                fig = px.pie(
                    wifi_counts,
                    values=wifi_counts.values,
                    names=wifi_counts.index,
                    title='Наличие Wi-Fi'
                )
                st.plotly_chart(fig, use_container_width=True)

        with col9:
            if 'bluetooth' in df.columns:
                bluetooth_counts = df['bluetooth'].value_counts()
                fig = px.pie(
                    bluetooth_counts,
                    values=bluetooth_counts.values,
                    names=bluetooth_counts.index,
                    title='Наличие Bluetooth'
                )
                st.plotly_chart(fig, use_container_width=True)

    with st.expander("Немного о данных"):
        st.write(f"Всего устройств: {len(df)}")
        st.write(f"Колонки с графиками: {len([col for col in df.columns if col != 'model'])} из {len(df.columns)}")
        st.write("Не визуализированы: model (название модеоей разное у всех производителей)")
# =========================================
# 3: ДИНАМИКА ВЫПУСКА
# =========================================
elif page == "Динамика выпуска":
    st.title("Динамика выпуска моделей по годам")

    st.markdown("""
    ### Анализ временных трендов
    """)

    st.subheader("Настройка интервала")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.markdown("От")
        use_min_year = st.checkbox("Задать минимальный год", value=True)
        if use_min_year:
            min_year = st.number_input(
                "от",
                min_value=int(df['release_year'].min()),
                max_value=int(df['release_year'].max()),
                value=int(df['release_year'].min()),
                key="min_year"
            )
        else:
            min_year = df['release_year'].min()
            st.info(f"Автоматически: {min_year}")

    with col2:
        st.markdown("До")
        use_max_year = st.checkbox("Задать максимальный год", value=True)
        if use_max_year:
            max_year = st.number_input(
                "до",
                min_value=int(df['release_year'].min()),
                max_value=int(df['release_year'].max()),
                value=int(df['release_year'].max()),
                key="max_year"
            )
        else:
            max_year = df['release_year'].max()
            st.info(f"Автоматически: {max_year}")

    with col3:
        st.markdown("Доп. настройки")

        selected_brands = st.multiselect(
            "Бренды",
            options=sorted(df['brand'].unique()),
            default=sorted(df['brand'].unique())[:5]
        )
        chart_type = st.radio(
            "Тип графика:",
            ["Линейный", "Столбчатый"],
            horizontal=True
        )

    filtered_df = df.copy()
    filtered_df = filtered_df[
        (filtered_df['release_year'] >= min_year) &
        (filtered_df['release_year'] <= max_year)
        ]

    if selected_brands:
        filtered_df = filtered_df[filtered_df['brand'].isin(selected_brands)]

    if not selected_brands:
        st.info("Надо выбрать производителя")
    else:
        draw_plot(filtered_df, min_year, max_year, chart_type)
        yearly_data = filtered_df.groupby(['release_year', 'brand']).size().reset_index(name='count')
        st.subheader(f"Статистика от {min_year} до {max_year}")

        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

        total_models = len(filtered_df)
        unique_brands = filtered_df['brand'].nunique()
        avg_models_per_year = yearly_data.groupby('release_year')['count'].sum().mean()
        most_productive_year = yearly_data.groupby('release_year')['count'].sum().idxmax()

        with col_stat1:
            st.metric("Всего моделей", f"{total_models:,}")

        with col_stat2:
            st.metric("Производителей", unique_brands)

        with col_stat3:
            st.metric("Среднее в год", f"{avg_models_per_year:.1f}")

        with col_stat4:
            st.metric("Максимальный год", most_productive_year)

        with st.expander("Детали"):
            pivot_table = yearly_data.pivot_table(
                index='brand',
                columns='release_year',
                values='count',
                fill_value=0
            ).astype(int)

            pivot_table['Всего'] = pivot_table.sum(axis=1)
            pivot_table.loc['Всего'] = pivot_table.sum()

            st.dataframe(pivot_table, use_container_width=True)
        st.markdown("---")
        with st.expander("Показать код"):
            with open("task3.py", "r", encoding="utf8") as file:
                code = file.read()
            st.markdown("### Код построения графика")
            st.code(code, language='python')

        st.markdown("---")
        st.markdown("Информация о выбранных данных:")
        st.write(f"- Общий период в датасете: {int(filtered_df['release_year'].min())}-{int(filtered_df['release_year'].max())}")
        st.write(f"- Уникальных брендов: {filtered_df['brand'].nunique()}")
        st.write(f"- Моделей в датасете: {len(filtered_df):,}")
# =========================================
# 4: ВОПРОС-ОТВЕТ
# =========================================
elif page == "Вопрос-ответ":
    st.title("Вопрос-ответ по датасету")

    st.markdown("""
    ### Форма для получения информации по датасету.
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Частые вопросы")
        st.markdown("Тыкни что бы увидеть ответ")

        if st.button("Сколько всего устройств в датасете?", key="q1"):
            total_devices = len(df)
            st.success(f"В датасете содержится {total_devices:,} устройств")

        if st.button("Какие типы устройств представлены?", key="q2"):
            device_types = df['device_type'].unique()
            device_types_str = ", ".join(device_types)
            st.success(f"Типы устройств: {device_types_str}")

        if st.button("Какой диапазон цен?", key="q3"):
            min_price = df['price'].min()
            max_price = df['price'].max()
            avg_price = df['price'].mean()
            st.success(f"""
            Диапазон цен:
            - Мин цена: ${min_price:,.2f}
            - Макс цена: ${max_price:,.2f}
            - Средняя цена: ${avg_price:,.2f}
            """)

        if st.button("Какие бренды самые популярные?", key="q4"):
            top_brands = df['brand'].value_counts().head(5)
            st.success("Топ-5 самых популярных брендов:")
            for brand, count in top_brands.items():
                st.write(f"- {brand}: {count} устройств")

    with col2:
        st.subheader("Еще запросики")
        st.markdown("Тут можно настроить запрос")

        st.markdown("Анализ по категориям")
        analysis_type = st.radio(
            "Тип анализа:",
            ["По брендам", "По типам устройств", "По операционным системам"],
            key="analysis_radio"
        )

        st.markdown("Фильтр по оперативной памяти")
        min_ram = st.slider(
            "Минимальный объем ОЗУ:",
            min_value=int(df['ram_gb'].min()),
            max_value=int(df['ram_gb'].max()),
            value=8,
            step=4,
            key="ram_slider"
        )

        st.markdown("Поиск по процессору")
        cpu_search = st.text_input(
            "Введите название процессора (например: Intel, AMD, Ryzen, i7):",
            placeholder="Intel i7",
            key="cpu_input"
        )

        st.markdown("Фильтр по цене")
        max_price_input = st.number_input(
            "Максимальная цена:",
            min_value=0,
            max_value=int(df['price'].max()) + 1000,
            value=2000,
            step=100,
            key="price_input"
        )

        if st.button("Применить фильтры и показать результаты", key="complex_filter"):
            with st.spinner("Анализируем..."):
                filtered_data = df.copy()
                filtered_data = filtered_data[filtered_data['ram_gb'] >= min_ram]
                filtered_data = filtered_data[filtered_data['price'] <= max_price_input]
                if cpu_search:
                    filtered_data = filtered_data[
                        filtered_data['cpu_model'].str.contains(cpu_search, case=False, na=False)
                    ]
                if analysis_type == "По брендам":
                    result = filtered_data['brand'].value_counts()
                    st.success(f"ОЗУ ≥ {min_ram}ГБ, цена ≤ ${max_price_input}:")
                    for brand, count in result.head(10).items():
                        st.write(f"- {brand}: {count} устройств")

                elif analysis_type == "По типам устройств":
                    result = filtered_data['device_type'].value_counts()
                    st.success(f"ОЗУ ≥ {min_ram}ГБ, цена ≤ ${max_price_input}")
                    for device_type, count in result.items():
                        st.write(f"- {device_type}: {count} устройств")

                else:
                    result = filtered_data['os'].value_counts()
                    st.success(f"ОЗУ ≥ {min_ram}ГБ, цена ≤ ${max_price_input}")
                    for os_name, count in result.items():
                        st.write(f"- {os_name}: {count} устройств")

                st.info(f"Всего найдно устройств: {len(filtered_data)}")
                if len(filtered_data) > 0:
                    avg_price_filtered = filtered_data['price'].mean()
                    st.info(f"Средняя цена в выборке: ${avg_price_filtered:,.2f}")
# =========================================
# 5: ПРОИЗВОДИТЕЛИ
# =========================================
elif page == "Производители":
    st.title("Производители и их штаб-квартиры")

    st.markdown("""
    ### Карта
    """)

    st.subheader("Производители в датасете")
    unique_brands = df['brand'].unique()
    brands_count = len(unique_brands)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.success(f"Всего уникальных производителей в датасете: {brands_count}")

        brands_per_row = 4
        brands_rows = [unique_brands[i:i + brands_per_row] for i in range(0, len(unique_brands), brands_per_row)]

        for row in brands_rows:
            cols = st.columns(brands_per_row)
            for i, brand in enumerate(row):
                with cols[i]:
                    st.info(f"{brand}")

    with col2:
        brand_stats = df['brand'].value_counts()
        top_5_brands = brand_stats.head(5)

        st.metric("Самый популярный", top_5_brands.index[0])
        st.metric(f"Устройств у {top_5_brands.index[0]}", top_5_brands.iloc[0])
        st.metric("Всего устройств", len(df))
    build_map(unique_brands, df, brand_stats)
    with st.expander("Показать код"):
        with open("task5.py", "r", encoding="utf8") as file:
            code = file.read()
        st.markdown("### Код построения графика")
        st.code(code, language='python')
# =========================================
# 6: ПРЕДСКАЗАНИЕ ЦЕН
# =========================================
elif page == "Предсказание цен":
    st.title("Предсказание цен на компьютеры")

    model_data = load_price_model(df)
    if model_data:
        st.subheader("Метрики модели")

        col_metric1, col_metric2, col_metric3 = st.columns(3)

        with col_metric1:
            st.metric(
                "Средняя абсолютная ошибка",
                f"${model_data['metrics']['mae']:.2f}"
            )

        with col_metric2:
            # так себе моделька вышла, но для теста сойдет
            st.metric(
                "R^2 Score",
                f"{model_data['metrics']['r2']:.3f}"
            )

        with col_metric3:
            st.metric(
                "Обучено на",
                f"{len(df):,}"
            )

        with st.expander("Важность характеристик в модели"):
            importances = model_data['feature_importances']
            sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

            for feature, importance in sorted_importances.items():
                feature_name = {
                    'brand': 'Бренд',
                    'device_type': 'Тип устройства',
                    'cpu_brand': 'Процессор',
                    'cpu_cores': 'Количество ядер',
                    'ram_gb': 'Оперативная память',
                    'storage_gb': 'Объем хранилища',
                    'gpu_brand': 'Видеокарта',
                    'display_size_in': 'Размер экрана'
                }.get(feature, feature)

                st.write(f"{feature_name}: {importance:.3f}")
                st.progress(importance)

        st.subheader("Предсказание цены")
        st.markdown("Введите параметры компьютера для предсказания его стоимости:")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                brand = st.selectbox(
                    "Бренд",
                    options=sorted(df['brand'].unique()),
                    help="Выберите производителя устройства"
                )

                device_type = st.selectbox(
                    "Тип устройства",
                    options=sorted(df['device_type'].unique()),
                    help="Выберите тип компьютерного устройства"
                )

                cpu_brand = st.selectbox(
                    "Процессор",
                    options=sorted(df['cpu_brand'].unique()),
                    help="Выберите производителя процессора"
                )

                gpu_brand = st.selectbox(
                    "Видеокарта",
                    options=sorted(df['gpu_brand'].unique()),
                    help="Выберите производителя видеокарты"
                )

            with col2:
                cpu_cores = st.slider(
                    "Количество ядер процессора",
                    min_value=int(df['cpu_cores'].min()),
                    max_value=int(df['cpu_cores'].max()),
                    value=4,
                    help="Выберите количество ядер процессора"
                )

                ram_gb = st.slider(
                    "Оперативная память (ГБ)",
                    min_value=int(df['ram_gb'].min()),
                    max_value=int(df['ram_gb'].max()),
                    value=8,
                    step=4,
                    help="Выберите объем оперативной памяти"
                )

                storage_gb = st.slider(
                    "Объем хранилища (ГБ)",
                    min_value=int(df['storage_gb'].min()),
                    max_value=int(df['storage_gb'].max()),
                    value=512,
                    step=128,
                    help="Выберите объем постоянной памяти"
                )

                screen_size = st.slider(
                    "Диагональ экрана (дюймы)",
                    min_value=float(df['display_size_in'].min()),
                    max_value=float(df['display_size_in'].max()),
                    value=15.6,
                    step=0.1,
                    help="Выберите размер экрана"
                )

            predict_button = st.form_submit_button(
                "Предсказать цену",
                use_container_width=True
            )

        if predict_button:
            try:
                input_data = {
                    'brand': brand,
                    'device_type': device_type,
                    'cpu_brand': cpu_brand,
                    'cpu_cores': cpu_cores,
                    'ram_gb': ram_gb,
                    'storage_gb': storage_gb,
                    'gpu_brand': gpu_brand,
                    'display_size_in': screen_size
                }
                input_df = pd.DataFrame([input_data])

                label_encoders = model_data['label_encoders']
                for col in ['brand', 'device_type', 'cpu_brand', 'gpu_brand']:
                    le = label_encoders[col]
                    if brand in le.classes_:
                        input_df[col] = le.transform([input_data[col]])[0]
                    else:
                        input_df[col] = 0

                input_df = input_df[model_data['feature_columns']]

                model = model_data['model']
                prediction = model.predict(input_df)[0]
                st.success(f"###Предсказанная цена: ${prediction:,.2f}")

                similar_devices = df[
                    (df['brand'] == brand) &
                    (df['device_type'] == device_type) &
                    (df['ram_gb'] >= ram_gb - 4) &
                    (df['ram_gb'] <= ram_gb + 4)
                    ]

                if not similar_devices.empty:
                    avg_price_similar = similar_devices['price'].mean()
                    price_diff = prediction - avg_price_similar

                    st.write(f"Средняя цена похожих устройств: ${avg_price_similar:,.2f}")

                    if price_diff > 0:
                        st.write(f"Устройство дороже на: ${price_diff:,.2f}")
                    else:
                        st.write(f"Устройство дешевле на: ${abs(price_diff):,.2f}")

            except Exception as e:
                st.error(f"Ошибка при предсказании: {e}")
    else:
        st.error("Брух")
# =========================================
# 7: ЧАТ С иишкой
# =========================================
elif page == "Чат с ИИ":
    st.title("Yandex GPT")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0

    def clear_chat_history():
        st.session_state.chat_history = []
        st.session_state.input_key += 1
        st.success("История очищена!")

    st.session_state.model =  "yandexgpt-lite"

    if st.button("Очистить историю", use_container_width=True):
        clear_chat_history()

    if not st.session_state.chat_history:
        st.info("пусто")
    else:
        for i, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                col_info1, col_info2 = st.columns([3, 1])
                with col_info1:
                    if "source" in message:
                        st.caption(f"{message['source']}")
                with col_info2:
                    if "timestamp" in message:
                        st.caption(f"{message['timestamp']}")
    question = st.text_area(
        "Ваш вопрос:",
        placeholder="Например: Как собрать игровой компьютер за 10000 рублей?",
        height=100,
        key=f"question_input_{st.session_state.input_key}"
    )
    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("Yandex GPT", use_container_width=True, type="primary"):
            timestamp = datetime.now().strftime("%H:%M:%S")

            st.session_state.chat_history.append({
                "role": "user",
                "content": question,
                "timestamp": timestamp
            })
            with st.spinner("Думаю над ответом..."):
                try:
                    # передаем историю из последних 2 сообщений что б модель была в теме
                    recent_history = st.session_state.chat_history[
                                     -2:] if st.session_state.chat_history else []

                    response = get_yandex_gpt_openai_response(question, recent_history)

                    response_source = f"Yandex GPT ({st.session_state.model})"

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "source": response_source,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                    st.session_state.input_key += 1
                except Exception as e:
                    st.error(f"Ошибка: {e}")
            st.session_state.input_key += 1
            st.rerun()

if __name__ == "__main__":
    st.sidebar.write("---")
    st.sidebar.markdown("""
    ### Информация
    ДЗ №4 
    [Исходники на Kaggle](https://www.kaggle.com/datasets/paperxd/all-computer-prices)
    """)
