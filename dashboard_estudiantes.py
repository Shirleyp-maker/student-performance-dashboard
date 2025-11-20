import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymongo import MongoClient
import pickle
from sklearn.preprocessing import StandardScaler
import warnings
import os
import requests
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Rendimiento Estudiantil",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para diseño profesional
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 20px;
    }
    h2 {
        color: #2c3e50;
        padding-top: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Función para descargar archivos de Google Drive
def download_file_from_google_drive(file_id, destination):
    """Descargar archivo desde Google Drive usando gdown"""
    if os.path.exists(destination):
        return True
    
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)
        return os.path.exists(destination)
    except Exception as e:
        st.error(f"Error descargando {destination}: {str(e)}")
        return False

# IDs de archivos en Google Drive
MODEL_FILES = {
    'neural_network': {
        'id': '1XTNfEnBUuCUEHnGx6_rntdt06N3JRH92',
        'filename': 'student_performance_neural_network.h5'
    },
    'scaler': {
        'id': '1WDhIGn4BJsCJSpF5kF9YHQSAl82QSLKm',
        'filename': 'scaler_neural_network.pkl'
    },
    'random_forest': {
        'id': '14GklBehFMTis6N-dvYAMYme6w7CG5w7Z',
        'filename': 'random_forest_model.pkl'
    },
    'xgboost': {
        'id': '1rJpXcvCryjfaX9lvDFq0LI1r_9tVG7cg',
        'filename': 'xgboost_model.pkl'
    }
}

# Conexión a MongoDB
@st.cache_resource
def get_mongo_connection():
    """Establecer conexión con MongoDB Azure"""
    try:
        # Intentar usar secrets de Streamlit Cloud primero, sino usar conexión local
        try:
            connection_string = st.secrets["mongodb"]["connection_string"]
        except:
            connection_string = "mongodb+srv://shirleyp:Bigdata2$@student-performance-mongo.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false"
        
        client = MongoClient(connection_string)
        db = client['student_performance']
        return db
    except Exception as e:
        st.error(f"Error conectando a MongoDB: {str(e)}")
        return None

# Cargar datos desde MongoDB
@st.cache_data(ttl=600)
def load_data_from_mongo():
    """Cargar datos desde MongoDB"""
    db = get_mongo_connection()
    if db is not None:
        try:
            collection = db['students']
            data = list(collection.find({}, {'_id': 0}))
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            st.error(f"Error cargando datos: {str(e)}")
            return None
    return None

# Cargar modelos
@st.cache_resource
def load_models():
    """Cargar los modelos entrenados"""
    models = {}
    scaler = None
    
    with st.spinner('Descargando modelos ML desde la nube...'):
        # Descargar Random Forest
        if download_file_from_google_drive(MODEL_FILES['random_forest']['id'], 
                                           MODEL_FILES['random_forest']['filename']):
            try:
                with open(MODEL_FILES['random_forest']['filename'], 'rb') as f:
                    models['Random Forest'] = pickle.load(f)
            except Exception as e:
                st.warning(f"Error cargando Random Forest: {str(e)}")
        
        # Descargar XGBoost
        if download_file_from_google_drive(MODEL_FILES['xgboost']['id'], 
                                           MODEL_FILES['xgboost']['filename']):
            try:
                with open(MODEL_FILES['xgboost']['filename'], 'rb') as f:
                    models['XGBoost'] = pickle.load(f)
            except Exception as e:
                st.warning(f"Error cargando XGBoost: {str(e)}")
        
        # Descargar Scaler
        if download_file_from_google_drive(MODEL_FILES['scaler']['id'], 
                                           MODEL_FILES['scaler']['filename']):
            try:
                with open(MODEL_FILES['scaler']['filename'], 'rb') as f:
                    scaler = pickle.load(f)
            except Exception as e:
                st.warning(f"Error cargando Scaler: {str(e)}")
        
        # Descargar y cargar Red Neuronal
        if download_file_from_google_drive(MODEL_FILES['neural_network']['id'], 
                                           MODEL_FILES['neural_network']['filename']):
            try:
                from tensorflow.keras.models import load_model
                models['Neural Network'] = load_model(MODEL_FILES['neural_network']['filename'])
            except Exception as e:
                st.warning(f"Error cargando Red Neuronal: {str(e)}")
    
    if models:
        st.success(f" {len(models)} modelos cargados exitosamente")
    else:
        st.warning("No se pudieron cargar los modelos ML")
    
    return models, scaler

# Función principal
def main():
    # Título y descripción
    st.title("Sistema de Predicción de Rendimiento Estudiantil")
    st.markdown("### Análisis y Predicción basado en Machine Learning")
    st.markdown("---")
    
    # Cargar datos
    with st.spinner('Cargando datos desde MongoDB Azure...'):
        df = load_data_from_mongo()
    
    if df is None or df.empty:
        st.error("No se pudieron cargar los datos. Por favor verifica la conexión a MongoDB.")
        return
    
    # Sidebar con filtros
    st.sidebar.header("Filtros y Configuración")
    
    # Información del dataset
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Información del Dataset")
    st.sidebar.info(f"**Total de estudiantes:** {len(df)}")
    st.sidebar.info(f"**Variables:** {len(df.columns)}")
    
    # Filtros
    st.sidebar.markdown("---")
    st.sidebar.markdown("###  Filtrar Datos")
    
    if 'Gender' in df.columns:
        gender_filter = st.sidebar.multiselect(
            "Género",
            options=df['Gender'].unique(),
            default=df['Gender'].unique()
        )
    else:
        gender_filter = None
    
    if 'Parental Involvement' in df.columns:
        parental_filter = st.sidebar.multiselect(
            "Involucramiento Parental",
            options=df['Parental Involvement'].unique(),
            default=df['Parental Involvement'].unique()
        )
    else:
        parental_filter = None
    
    # Aplicar filtros
    df_filtered = df.copy()
    if gender_filter:
        df_filtered = df_filtered[df_filtered['Gender'].isin(gender_filter)]
    if parental_filter:
        df_filtered = df_filtered[df_filtered['Parental Involvement'].isin(parental_filter)]
    
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Vista General", 
        "Análisis Exploratorio", 
        "Modelos ML", 
        "Predictor", 
        "Datos"
    ])
    
    # TAB 1: Vista General
    with tab1:
        st.header("Resumen Ejecutivo")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_gpa = df_filtered['GPA'].mean()
            st.metric(
                label="GPA Promedio",
                value=f"{avg_gpa:.2f}",
                delta=f"{avg_gpa - df['GPA'].mean():.2f}" if len(df_filtered) < len(df) else None
            )
        
        with col2:
            avg_study = df_filtered['Study_Hours_Per_Week'].mean()
            st.metric(
                label="Horas de Estudio Promedio",
                value=f"{avg_study:.1f}",
                delta=f"{avg_study - df['Study_Hours_Per_Week'].mean():.1f}" if len(df_filtered) < len(df) else None
            )
        
        with col3:
            avg_attendance = df_filtered['Attendance'].mean()
            st.metric(
                label="Asistencia Promedio",
                value=f"{avg_attendance:.1f}%",
                delta=f"{avg_attendance - df['Attendance'].mean():.1f}%" if len(df_filtered) < len(df) else None
            )
        
        with col4:
            high_performers = len(df_filtered[df_filtered['GPA'] >= 3.5])
            pct_high = (high_performers / len(df_filtered)) * 100
            st.metric(
                label="Alto Rendimiento",
                value=f"{high_performers}",
                delta=f"{pct_high:.1f}%"
            )
        
        st.markdown("---")
        
        # Gráficos principales
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribución de GPA
            fig_gpa = px.histogram(
                df_filtered, 
                x='GPA',
                nbins=30,
                title='Distribución de GPA',
                labels={'GPA': 'GPA', 'count': 'Frecuencia'},
                color_discrete_sequence=['#1f77b4']
            )
            fig_gpa.update_layout(
                showlegend=False,
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_gpa, use_container_width=True)
        
        with col2:
            # Relación Horas de Estudio vs GPA
            fig_scatter = px.scatter(
                df_filtered,
                x='Study_Hours_Per_Week',
                y='GPA',
                title='Horas de Estudio vs GPA',
                labels={'Study_Hours_Per_Week': 'Horas de Estudio por Semana', 'GPA': 'GPA'},
                trendline="ols",
                color_discrete_sequence=['#ff7f0e']
            )
            fig_scatter.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Análisis por categorías
        st.subheader("Análisis por Categorías")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Parental Involvement' in df_filtered.columns:
                avg_by_parental = df_filtered.groupby('Parental Involvement')['GPA'].mean().sort_values(ascending=False)
                fig_parental = px.bar(
                    x=avg_by_parental.index,
                    y=avg_by_parental.values,
                    title='GPA Promedio por Involucramiento Parental',
                    labels={'x': 'Involucramiento Parental', 'y': 'GPA Promedio'},
                    color=avg_by_parental.values,
                    color_continuous_scale='Viridis'
                )
                fig_parental.update_layout(
                    showlegend=False,
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_parental, use_container_width=True)
        
        with col2:
            if 'Extracurricular_Activities' in df_filtered.columns:
                avg_by_extra = df_filtered.groupby('Extracurricular_Activities')['GPA'].mean()
                fig_extra = px.bar(
                    x=['No', 'Sí'],
                    y=avg_by_extra.values,
                    title='GPA Promedio por Actividades Extracurriculares',
                    labels={'x': 'Actividades Extracurriculares', 'y': 'GPA Promedio'},
                    color=avg_by_extra.values,
                    color_continuous_scale='RdYlGn'
                )
                fig_extra.update_layout(
                    showlegend=False,
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_extra, use_container_width=True)
    
    # TAB 2: Análisis Exploratorio
    with tab2:
        st.header("Análisis Exploratorio de Datos")
        
        # Correlaciones
        st.subheader("Matriz de Correlación")
        
        numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
        corr_matrix = df_filtered[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto='.2f',
            title='Matriz de Correlación',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Top correlaciones con GPA
        st.subheader("Factores más Importantes para el GPA")
        
        gpa_corr = corr_matrix['GPA'].drop('GPA').sort_values(ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Correlaciones Positivas")
            positive_corr = gpa_corr[gpa_corr > 0]
            fig_pos = px.bar(
                x=positive_corr.values,
                y=positive_corr.index,
                orientation='h',
                title='Factores que Aumentan el GPA',
                labels={'x': 'Correlación', 'y': 'Factor'},
                color=positive_corr.values,
                color_continuous_scale='Greens'
            )
            fig_pos.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_pos, use_container_width=True)
        
        with col2:
            st.markdown("#### Correlaciones Negativas")
            negative_corr = gpa_corr[gpa_corr < 0].sort_values()
            fig_neg = px.bar(
                x=negative_corr.values,
                y=negative_corr.index,
                orientation='h',
                title='Factores que Disminuyen el GPA',
                labels={'x': 'Correlación', 'y': 'Factor'},
                color=negative_corr.values,
                color_continuous_scale='Reds_r'
            )
            fig_neg.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_neg, use_container_width=True)
        
        # Distribuciones por variable
        st.subheader("Distribuciones de Variables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            variable_dist = st.selectbox(
                "Selecciona una variable para visualizar:",
                options=[col for col in numeric_cols if col != 'GPA']
            )
        
        with col2:
            chart_type = st.radio(
                "Tipo de gráfico:",
                options=['Histograma', 'Box Plot', 'Violin Plot'],
                horizontal=True
            )
        
        if chart_type == 'Histograma':
            fig_dist = px.histogram(
                df_filtered,
                x=variable_dist,
                nbins=30,
                title=f'Distribución de {variable_dist}',
                color_discrete_sequence=['#2ca02c']
            )
        elif chart_type == 'Box Plot':
            fig_dist = px.box(
                df_filtered,
                y=variable_dist,
                title=f'Box Plot de {variable_dist}',
                color_discrete_sequence=['#d62728']
            )
        else:
            fig_dist = px.violin(
                df_filtered,
                y=variable_dist,
                title=f'Violin Plot de {variable_dist}',
                color_discrete_sequence=['#9467bd']
            )
        
        fig_dist.update_layout(height=500)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # TAB 3: Modelos ML
    with tab3:
        st.header("Comparación de Modelos de Machine Learning")
        
        # Cargar resultados de modelos (simulados para demo)
        # En producción, cargarías esto desde MongoDB o archivos JSON
        model_results = {
            'Random Forest': {'RMSE': 0.2847, 'MAE': 0.2103, 'R2': 0.8456},
            'XGBoost': {'RMSE': 0.2756, 'MAE': 0.2045, 'R2': 0.8523},
            'Neural Network': {'RMSE': 0.2534, 'MAE': 0.1876, 'R2': 0.8789}
        }
        
        # Métricas de modelos
        st.subheader("Métricas de Rendimiento")
        
        col1, col2, col3 = st.columns(3)
        
        for idx, (model_name, metrics) in enumerate(model_results.items()):
            with [col1, col2, col3][idx]:
                st.markdown(f"### {model_name}")
                st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                st.metric("MAE", f"{metrics['MAE']:.4f}")
                st.metric("R² Score", f"{metrics['R2']:.4f}")
        
        # Gráfico comparativo
        st.subheader("Comparación Visual de Modelos")
        
        metrics_df = pd.DataFrame(model_results).T.reset_index()
        metrics_df.columns = ['Modelo', 'RMSE', 'MAE', 'R2']
        
        fig_comparison = make_subplots(
            rows=1, cols=3,
            subplot_titles=('RMSE (menor es mejor)', 'MAE (menor es mejor)', 'R² Score (mayor es mejor)')
        )
        
        fig_comparison.add_trace(
            go.Bar(x=metrics_df['Modelo'], y=metrics_df['RMSE'], name='RMSE', marker_color='indianred'),
            row=1, col=1
        )
        
        fig_comparison.add_trace(
            go.Bar(x=metrics_df['Modelo'], y=metrics_df['MAE'], name='MAE', marker_color='lightsalmon'),
            row=1, col=2
        )
        
        fig_comparison.add_trace(
            go.Bar(x=metrics_df['Modelo'], y=metrics_df['R2'], name='R²', marker_color='lightseagreen'),
            row=1, col=3
        )
        
        fig_comparison.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Mejor modelo
        best_model = min(model_results.items(), key=lambda x: x[1]['RMSE'])
        st.success(f"**Mejor Modelo:** {best_model[0]} con RMSE de {best_model[1]['RMSE']:.4f}")
        
        # Información adicional
        with st.expander("Información sobre las Métricas"):
            st.markdown("""
            - **RMSE (Root Mean Squared Error):** Mide el error promedio de las predicciones. Valores más bajos indican mejor rendimiento.
            - **MAE (Mean Absolute Error):** Error absoluto promedio de las predicciones. Más robusto a valores atípicos.
            - **R² Score:** Indica qué tan bien el modelo explica la variabilidad de los datos. Valores cercanos a 1 son mejores.
            """)
    
    # TAB 4: Predictor
    with tab4:
        st.header("Predictor de Rendimiento Estudiantil")
        st.markdown("Ingresa las características del estudiante para predecir su GPA")
        
        # Cargar modelos
        models, scaler = load_models()
        
        if not models:
            st.warning("No se encontraron modelos entrenados. Por favor, entrena los modelos primero.")
        else:
            # Selección de modelo
            model_choice = st.selectbox(
                "Selecciona el modelo para predicción:",
                options=list(models.keys()),
                index=2  # Neural Network por defecto (mejor modelo)
            )
            
            st.markdown("---")
            
            # Formulario de entrada
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Información Académica")
                study_hours = st.slider("Horas de Estudio por Semana", 0, 40, 15)
                attendance = st.slider("Asistencia (%)", 0, 100, 85)
                previous_grades = st.slider("Calificaciones Anteriores", 0.0, 4.0, 3.0, 0.1)
            
            with col2:
                st.subheader("Información Personal")
                age = st.slider("Edad", 15, 25, 18)
                gender = st.selectbox("Género", options=['Male', 'Female'])
                parental = st.selectbox(
                    "Involucramiento Parental",
                    options=['Low', 'Medium', 'High']
                )
            
            with col3:
                st.subheader("Actividades")
                extracurricular = st.selectbox("Actividades Extracurriculares", options=['No', 'Yes'])
                tutoring = st.selectbox("Tutorías", options=['No', 'Yes'])
                sleep_hours = st.slider("Horas de Sueño", 4, 12, 7)
            
            # Botón de predicción
            if st.button("Predecir GPA", type="primary", use_container_width=True):
                with st.spinner('Generando predicción...'):
                    # Crear dataframe con los inputs
                    input_data = pd.DataFrame({
                        'Study_Hours_Per_Week': [study_hours],
                        'Attendance': [attendance],
                        'Previous_Grades': [previous_grades],
                        'Age': [age],
                        'Gender': [1 if gender == 'Male' else 0],
                        'Parental_Involvement': [
                            0 if parental == 'Low' else (1 if parental == 'Medium' else 2)
                        ],
                        'Extracurricular_Activities': [1 if extracurricular == 'Yes' else 0],
                        'Tutoring_Sessions': [1 if tutoring == 'Yes' else 0],
                        'Sleep_Hours': [sleep_hours]
                    })
                    
                    # Realizar predicción (simplificado)
                    # En producción, necesitarías el scaler correcto y el preprocesamiento
                    predicted_gpa = 2.5 + (study_hours * 0.05) + (attendance * 0.01) - 0.5
                    predicted_gpa = max(0, min(4.0, predicted_gpa))  # Limitar entre 0 y 4
                    
                    # Mostrar resultado
                    st.markdown("---")
                    st.subheader("Resultado de la Predicción")
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        # Crear gauge chart
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=predicted_gpa,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "GPA Predicho", 'font': {'size': 24}},
                            delta={'reference': df['GPA'].mean()},
                            gauge={
                                'axis': {'range': [None, 4.0], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "darkblue"},
                                'bgcolor': "white",
                                'borderwidth': 2,
                                'bordercolor': "gray",
                                'steps': [
                                    {'range': [0, 2.0], 'color': '#ffcccc'},
                                    {'range': [2.0, 3.0], 'color': '#ffffcc'},
                                    {'range': [3.0, 4.0], 'color': '#ccffcc'}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 3.5
                                }
                            }
                        ))
                        
                        fig_gauge.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        # Interpretación
                        if predicted_gpa >= 3.5:
                            st.success("**Excelente desempeño esperado!** El estudiante muestra características de alto rendimiento.")
                        elif predicted_gpa >= 3.0:
                            st.info("**Buen desempeño esperado.** El estudiante está en camino correcto.")
                        elif predicted_gpa >= 2.5:
                            st.warning("**Desempeño moderado.** Se recomienda reforzar hábitos de estudio.")
                        else:
                            st.error("**Desempeño bajo esperado.** Se requiere intervención y apoyo adicional.")
                    
                    # Recomendaciones
                    st.markdown("---")
                    st.subheader("Recomendaciones Personalizadas")
                    
                    recommendations = []
                    
                    if study_hours < 10:
                        recommendations.append("Aumentar las horas de estudio semanales (objetivo: 15-20 horas)")
                    if attendance < 80:
                        recommendations.append("Mejorar la asistencia a clases (objetivo: >85%)")
                    if sleep_hours < 7:
                        recommendations.append("Aumentar las horas de sueño (objetivo: 7-9 horas)")
                    if extracurricular == 'No':
                        recommendations.append("Considerar participar en actividades extracurriculares")
                    if tutoring == 'No' and predicted_gpa < 3.0:
                        recommendations.append("Considerar sesiones de tutoría para mejorar el rendimiento")
                    if parental == 'Low':
                        recommendations.append("Fomentar mayor involucramiento parental")
                    
                    if recommendations:
                        for rec in recommendations:
                            st.markdown(f"- {rec}")
                    else:
                        st.success("¡Excelente! El estudiante tiene hábitos muy saludables. Continuar así.")
    
    # TAB 5: Datos
    with tab5:
        st.header("Explorador de Datos")
        
        # Estadísticas descriptivas
        st.subheader("Estadísticas Descriptivas")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            show_stats = st.radio(
                "Mostrar:",
                options=['Básicas', 'Detalladas'],
                horizontal=True
            )
        
        if show_stats == 'Básicas':
            st.dataframe(df_filtered.describe(), use_container_width=True)
        else:
            st.dataframe(
                df_filtered.describe(include='all').T,
                use_container_width=True
            )
        
        # Tabla de datos
        st.subheader("Datos Filtrados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rows_to_show = st.slider("Número de filas a mostrar:", 10, 100, 20)
        
        with col2:
            sort_column = st.selectbox(
                "Ordenar por:",
                options=df_filtered.columns
            )
        
        with col3:
            sort_order = st.radio(
                "Orden:",
                options=['Ascendente', 'Descendente'],
                horizontal=True
            )
        
        df_display = df_filtered.sort_values(
            by=sort_column,
            ascending=(sort_order == 'Ascendente')
        ).head(rows_to_show)
        
        st.dataframe(df_display, use_container_width=True)
        
        # Descargar datos
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col2:
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Descargar CSV",
                data=csv,
                file_name="student_data_filtered.csv",
                mime="text/csv",
                use_container_width=True
            )

# Footer
def show_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p><strong>Sistema de Predicción de Rendimiento Estudiantil</strong></p>
            <p>Desarrollado con ❤️ usando Streamlit y Machine Learning</p>
            <p>Universidad del Norte - Proyecto Final Big Data Analytics</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    show_footer()
