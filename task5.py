import json

import streamlit as st
import pandas as pd
import plotly.express as px

def get_fallback_company_info(brand_name, brand_stats):
    # если что-то сломаетчя
    fallback_data = {
        'Dell': {'lat': 30.2672, 'lon': -97.7431, 'city': 'Раунд-Рок, Техас, США'},
        'HP': {'lat': 37.3541, 'lon': -121.9552, 'city': 'Пало-Альто, Калифорния, США'},
        'Lenovo': {'lat': 39.9042, 'lon': 116.4074, 'city': 'Пекин, Китай'},
        'Apple': {'lat': 37.3349, 'lon': -122.0090, 'city': 'Купертино, Калифорния, США'},
        'Asus': {'lat': 25.0172, 'lon': 121.3687, 'city': 'Тайбэй, Тайвань'},
        'Acer': {'lat': 25.0172, 'lon': 121.3687, 'city': 'Тайбэй, Тайвань'},
        'MSI': {'lat': 25.0172, 'lon': 121.3687, 'city': 'Тайбэй, Тайвань'},
        'Samsung': {'lat': 37.5665, 'lon': 126.9780, 'city': 'Сеул, Южная Корея'},
    }

    info = {
        'brand': brand_name,
        'devices_in_dataset': brand_stats.get(brand_name, 0),
        'found_via_api': False
    }

    if brand_name in fallback_data:
        info.update(fallback_data[brand_name])
    else:
        import random
        info.update({
            'lat': random.uniform(30, 50),
            'lon': random.uniform(-120, 140),
            'city': 'Местоположение неизвестно'
        })

    return info


@st.cache_data(ttl=3600)
def get_company_info(brand_name, brand_stats):
    try:
        import requests
        import time

        base_info = {
            'brand': brand_name,
            'devices_in_dataset': brand_stats.get(brand_name, 0)
        }

        search_query = f"{brand_name}"
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            'q': search_query,
            'limit': 1,
            'format': 'json'
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Referer': 'https://openstreetmap.org'
        }
        response = requests.get(url, params=params, headers=headers)

        print(brand_name)

        time.sleep(1)

        if response.status_code == 200:
            data = response.json()
            print(data)
            if data:
                location = data[0]
                base_info.update({
                    'lat': float(location['lat']),
                    'lon': float(location['lon']),
                    'city': location.get('display_name', 'Неизвестно'),
                    'found_via_api': True
                })
                return base_info

        return get_fallback_company_info(brand_name, brand_stats)

    except Exception as e:
        st.error(f"Ошибка: {e}")
        return get_fallback_company_info(brand_name, brand_stats)
def build_map(unique_brands, df, brand_stats):
    st.subheader("Штаб-квартиры производителей")

    with st.spinner("Получаем местоположение..."):
        manufacturers_data = []
        progress_bar = st.progress(0)

        for i, brand in enumerate(unique_brands[:10]):
            brand_info = get_company_info(brand, brand_stats)
            manufacturers_data.append(brand_info)
            progress_bar.progress((i + 1) / min(10, len(unique_brands)))

    map_df = pd.DataFrame(manufacturers_data)

    if not map_df.empty:
        st.subheader("Карта штаб-квартир")

        api_success_count = map_df['found_via_api'].sum()
        st.info(f"Данные получены через API для {api_success_count} из {len(map_df)} производителей")

        col_map1, col_map2 = st.columns([3, 1])

        with col_map2:
            st.markdown("Настройки карты")
            map_style = st.selectbox(
                "Стиль карты:",
                ["open-street-map", "carto-darkmatter"],
                index=0
            )


        with col_map1:
            display_df = map_df
            display_df = display_df[display_df['found_via_api'] == True]
            if display_df.empty:
                st.warning("Нет данных, полученных через API")
                display_df = map_df


            fig = px.scatter_mapbox(
                display_df,
                lat="lat",
                lon="lon",
                hover_name="brand",
                hover_data={
                    'lat': False,
                    'lon': False,
                    'brand': True,
                    'city': True,
                    'devices_in_dataset': True,
                    'found_via_api': True
                },
                custom_data=['brand', 'city', 'devices_in_dataset', 'found_via_api'],
                size=None,
                size_max= 20,
                color="found_via_api",
                color_discrete_map={True: '#2E86AB', False: '#A23B72'},
                zoom=1,
                height=500,
                title="Географическое расположение штаб-квартир производителей"
            )



            fig.update_layout(mapbox_style=map_style)
            fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0})
            fig.update_layout(legend_title_text="Данные с API")

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Информация о производителях")

        search_brand = st.text_input("Поиск производителя:", placeholder="Введите название бренда...")

        display_manufacturers = map_df
        if search_brand:
            display_manufacturers = display_manufacturers[
                display_manufacturers['brand'].str.contains(search_brand, case=False, na=False)
            ]

        for _, manufacturer in display_manufacturers.iterrows():
            with st.expander(f"{manufacturer['brand']} - {manufacturer['city'].split(',')[0]}"):
                col_info1, col_info2 = st.columns(2)

                with col_info1:
                    st.write(f"Местоположение: {manufacturer['city']}")
                    st.write(f"Устройств в датасете: {manufacturer['devices_in_dataset']}")

                    data_source = "API" if manufacturer['found_via_api'] else "Резервные данные"
                    st.write(f"Источник данных: {data_source}")

                with col_info2:
                    brand_models = df[df['brand'] == manufacturer['brand']]
                    if not brand_models.empty:
                        st.write("Статистика")
                        st.write(f"Средняя цена {brand_models['price'].mean():,.2f}")
                        st.write(f"Популярный тип {brand_models['device_type'].mode().iloc[0]}")
                        st.write(
                            f"Годы выпуска: {brand_models['release_year'].min()}-{brand_models['release_year'].max()}")
    else:
        st.error("эх]'")