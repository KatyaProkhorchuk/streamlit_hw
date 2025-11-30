import streamlit as st
import plotly.express as px

def draw_plot(filtered_df, min_year, max_year, chart_type):

    if filtered_df.empty:
        st.warning("Нет данных")
    else:
        yearly_data = filtered_df.groupby(['release_year', 'brand']).size().reset_index(name='count')
        st.subheader("Динамика выпуска компухтеров")

        if chart_type == "Линейный":
            fig = px.line(
                yearly_data,
                x='release_year',
                y='count',
                color='brand',
                title=f'Динамика по годам ({min_year}-{max_year})',
                labels={
                    'release_year': 'Год',
                    'count': 'Количество',
                    'brand': 'Производитель'
                },
                markers=True
            )
        else:
            fig = px.bar(
                yearly_data,
                x='release_year',
                y='count',
                color='brand',
                title=f'Динамика по годам ({min_year}-{max_year})',
                labels={
                    'release_year': 'Год',
                    'count': 'Количество',
                    'brand': 'Производитель'
                },
                barmode='group'
            )

        fig.update_layout(
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        st.plotly_chart(fig, use_container_width=True)
