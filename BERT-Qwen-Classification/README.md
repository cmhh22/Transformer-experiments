# BERT-Qwen Text Classification

## 📋 Descripción

Proyecto de clasificación de texto utilizando modelos Transformer (BERT y Qwen). Implementa fine-tuning de modelos preentrenados para tareas de clasificación con análisis comparativo de rendimiento.

## 🎯 Objetivos

- Fine-tuning de BERT para clasificación de texto
- Implementación de Qwen para tareas de NLP
- Análisis comparativo de modelos Transformer
- Evaluación exhaustiva con métricas avanzadas
- Visualización de resultados y embeddings

## 🚀 Características

- **Modelos implementados**: BERT-base, Qwen
- **Arquitectura**: PyTorch + Transformers (HuggingFace)
- **Técnicas**: Fine-tuning, Transfer Learning, Attention Analysis
- **Evaluación**: Accuracy, F1-Score, Confusion Matrix, ROC-AUC
- **Visualizaciones**: Embeddings t-SNE, Attention Heatmaps

## 📁 Estructura del Proyecto

```
BERT-Qwen-Classification/
├── main.py                 # Script principal de entrenamiento
├── test_quick.py           # Tests rápidos
├── requirements.txt        # Dependencias
├── README.md              # Documentación
├── data/                  # Datasets
│   ├── raw/              # Datos originales
│   └── processed/        # Datos procesados
├── models/               # Modelos guardados
├── notebooks/            # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_bert_training.ipynb
│   └── 03_qwen_training.ipynb
└── src/                  # Código fuente
    ├── __init__.py
    ├── config.py         # Configuración
    ├── model.py          # Arquitecturas de modelos
    ├── data_loader.py    # Carga de datos
    ├── train.py          # Lógica de entrenamiento
    ├── evaluate.py       # Evaluación
    └── utils.py          # Utilidades
```

## 🛠️ Instalación

```bash
# Clonar el repositorio
git clone https://github.com/cmhh22/transformer-experiments.git
cd transformer-experiments/BERT-Qwen-Classification

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## 📊 Uso

### Entrenamiento

```bash
# Entrenar modelo BERT
python main.py --model bert --epochs 10 --batch_size 32

# Entrenar modelo Qwen
python main.py --model qwen --epochs 10 --batch_size 16
```

### Evaluación

```bash
# Evaluar modelo guardado
python main.py --mode eval --model_path models/best_model.pth
```

### Tests rápidos

```bash
python test_quick.py
```

## 📈 Resultados

Los resultados y métricas se guardan en:
- `models/`: Modelos entrenados y checkpoints
- `notebooks/`: Análisis detallado y visualizaciones
- Métricas: Accuracy, F1-Score, Precision, Recall

## 🔧 Configuración

Ajusta los hiperparámetros en `src/config.py`:

```python
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
MAX_LENGTH = 512
NUM_EPOCHS = 10
```

## 📝 Dataset

El proyecto soporta diversos datasets de clasificación de texto:
- IMDB Reviews
- AG News
- Custom datasets (CSV/JSON)

## 🎓 Conceptos Clave

- **Transfer Learning**: Aprovechamiento de modelos preentrenados
- **Attention Mechanism**: Mecanismo de atención en Transformers
- **Tokenization**: Procesamiento de texto con tokenizers específicos
- **Fine-tuning**: Ajuste de modelos preentrenados

## 📚 Referencias

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

## 👤 Autor

**Carlos Hernández**
- GitHub: [@cmhh22](https://github.com/cmhh22)
- LinkedIn: [Carlos Hernández](https://linkedin.com/in/cmhh22)

## 📄 Licencia

MIT License - ver archivo [LICENSE](LICENSE) para detalles.

## 🚧 Estado del Proyecto

🚀 En desarrollo activo - Enero 2026
