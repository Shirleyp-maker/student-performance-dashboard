# 🎓 Dashboard de Predicción de Rendimiento Estudiantil

Sistema de análisis y predicción de rendimiento estudiantil usando Machine Learning y MongoDB Azure.

## 📋 Características

- **Conexión a MongoDB Azure**: Carga automática de datos desde la nube
- **Análisis Exploratorio**: Visualizaciones interactivas de datos
- **Modelos ML**: Comparación de Random Forest, XGBoost y Redes Neuronales
- **Predictor Interactivo**: Predicción en tiempo real del GPA
- **Filtros Dinámicos**: Exploración personalizada de datos
- **Diseño Profesional**: Interfaz moderna y responsive

## 🚀 Instalación

### 1. Requisitos Previos
- Python 3.9 o superior
- Visual Studio Code (recomendado)
- Git

### 2. Clonar o Descargar el Proyecto
```bash
# Si tienes Git
git clone [URL_DEL_REPOSITORIO]
cd proyecto2

# O simplemente copia los archivos en tu carpeta del proyecto
```

### 3. Crear Entorno Virtual (Recomendado)
```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Instalar Dependencias
```bash
pip install -r requirements.txt
```

## 📂 Estructura del Proyecto

```
proyecto2/
│
├── dashboard_estudiantes.py    # Dashboard principal
├── requirements.txt            # Dependencias
├── README.md                   # Este archivo
│
├── modelos/                    # Carpeta de modelos (crear si no existe)
│   ├── neural_network_model.h5
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
│
└── data/                       # Datos locales (opcional)
    └── student_data.csv
```

## ⚙️ Configuración

### MongoDB Azure
El dashboard se conecta automáticamente a MongoDB Azure usando las credenciales configuradas. Si necesitas cambiar la conexión, edita la función `get_mongo_connection()` en el archivo `dashboard_estudiantes.py`:

```python
connection_string = "mongodb+srv://usuario:contraseña@cluster.mongodb.net/..."
```

### Modelos ML
Asegúrate de que los archivos de modelos estén en la ubicación correcta:
- `neural_network_model.h5`
- `random_forest_model.pkl`
- `xgboost_model.pkl`

## 🎯 Ejecutar el Dashboard

### Opción 1: Desde la Terminal
```bash
streamlit run dashboard_estudiantes.py
```

### Opción 2: Desde Visual Studio Code
1. Abre el proyecto en VS Code
2. Abre la terminal integrada (Ctrl + `)
3. Ejecuta: `streamlit run dashboard_estudiantes.py`

El dashboard se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📱 Uso del Dashboard

### 1. Vista General
- Visualiza métricas clave del dataset
- Explora distribuciones y correlaciones
- Analiza patrones generales

### 2. Análisis Exploratorio
- Matriz de correlación interactiva
- Factores más importantes para el GPA
- Distribuciones por variable

### 3. Modelos ML
- Compara el rendimiento de los 3 modelos
- Visualiza métricas (RMSE, MAE, R²)
- Identifica el mejor modelo

### 4. Predictor
- Ingresa características del estudiante
- Obtén predicción de GPA en tiempo real
- Recibe recomendaciones personalizadas

### 5. Datos
- Explora el dataset completo
- Aplica filtros y ordenamientos
- Descarga datos procesados

## 🎨 Personalización

### Cambiar Colores
Edita la sección de CSS en `dashboard_estudiantes.py`:
```python
st.markdown("""
    <style>
    /* Personaliza aquí */
    </style>
""", unsafe_allow_html=True)
```

### Agregar Nuevas Visualizaciones
1. Crea nuevas funciones de visualización
2. Agrégalas en las tabs correspondientes
3. Usa Plotly para gráficos interactivos

## 🔧 Solución de Problemas

### Error de Conexión a MongoDB
- Verifica tu conexión a internet
- Confirma las credenciales en el código
- Revisa los permisos del cluster

### Error al Cargar Modelos
- Asegúrate de que los archivos .pkl y .h5 existan
- Verifica las rutas en el código
- Reinstala las librerías si es necesario

### Error de Dependencias
```bash
# Reinstalar todas las dependencias
pip install --upgrade -r requirements.txt
```

### Puerto Ocupado
Si el puerto 8501 está ocupado:
```bash
streamlit run dashboard_estudiantes.py --server.port 8502
```

## 📊 Métricas de los Modelos

| Modelo | RMSE | MAE | R² Score |
|--------|------|-----|----------|
| Random Forest | 0.2847 | 0.2103 | 0.8456 |
| XGBoost | 0.2756 | 0.2045 | 0.8523 |
| **Neural Network** | **0.2534** | **0.1876** | **0.8789** |

*La Red Neuronal es el mejor modelo según las métricas.*

## 🤝 Contribuciones

Este proyecto fue desarrollado como parte del curso de Big Data Analytics de la Universidad del Norte.

**Equipo:**
- Shirley Padilla
- Johanna Blanquicet
- David Florez

## 📝 Notas Adicionales

- El dashboard utiliza caché para optimizar el rendimiento
- Los datos se actualizan cada 10 minutos desde MongoDB
- Las predicciones son aproximadas y con fines educativos
- Se recomienda usar Chrome o Firefox para mejor experiencia

## 📧 Soporte

Si tienes problemas o preguntas:
1. Revisa la sección de Solución de Problemas
2. Verifica los logs en la terminal
3. Contacta al equipo de desarrollo

## 🎓 Licencia

Proyecto educativo - Universidad del Norte © 2024

---

**¡Disfruta explorando los datos y predicciones! 🚀**
