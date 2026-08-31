# Predicción de Cancelación de Clientes (Churn) — Interconnect

Proyecto final del bootcamp de Ciencia de Datos (TripleTen). Desarrollo de un modelo de clasificación que predice si un cliente de la operadora de telecomunicaciones **Interconnect** cancelará su servicio, con el fin de que el equipo de marketing pueda ofrecer códigos promocionales o planes especiales a los clientes en riesgo de abandono.

## 📋 Contexto del negocio

Interconnect ofrece servicios de telefonía fija e internet (DSL y fibra óptica), además de servicios adicionales como seguridad en línea, soporte técnico, backup en la nube y streaming de TV/películas. El objetivo es anticipar la cancelación (**churn**) a partir de la información contractual, demográfica y de servicios de cada cliente.

- **Variable objetivo:** `EndDate == 'No'` → cliente activo (0) / cliente cancelado (1)
- **Métrica principal:** AUC-ROC
- **Métrica secundaria:** Exactitud (Accuracy)

## 🗂️ Datos

El proyecto integra 4 fuentes de datos, unidas por `customerID`:

| Archivo | Contenido |
|---|---|
| `contract.csv` | Información contractual (tipo de contrato, fechas, método de pago, cargos) |
| `personal.csv` | Datos demográficos del cliente |
| `internet.csv` | Servicios de internet contratados |
| `phone.csv` | Servicios telefónicos contratados |

## 🔍 Metodología

1. **Preprocesamiento:** unión de las 4 tablas, construcción del target a partir de `EndDate`, cálculo de antigüedad del cliente (respecto a la fecha de referencia del proyecto, 1 de febrero de 2020), tratamiento de nulos estructurales y codificación de variables categóricas (binaria + One-Hot).
2. **Análisis exploratorio (EDA):** distribución del target, relación de variables numéricas y categóricas con el churn, matriz de correlación — excluyendo explícitamente las columnas derivadas de `EndDate` para evitar fuga de información.
3. **Modelos base:** comparación con validación cruzada estratificada (5 folds) de Logistic Regression, Random Forest, LightGBM y CatBoost.
4. **Ajuste de hiperparámetros:** `GridSearchCV` aplicado a los dos candidatos con mejor desempeño base (CatBoost y LightGBM), manejando el desbalance de clases (~26.5% churn) mediante `scale_pos_weight`.
5. **Evaluación final:** selección automática del mejor modelo por AUC-ROC en validación cruzada, evaluado sobre un conjunto de prueba nunca antes visto.

## 📊 Resultados

| Modelo | AUC-ROC (CV) |
|---|---|
| Logistic Regression | 0.8471 |
| Random Forest | 0.8275 |
| LightGBM (ajustado) | 0.8648 |
| **CatBoost (ajustado)** | **0.8685** |

**Modelo final: CatBoost**

| Métrica (conjunto de prueba) | Valor |
|---|---|
| AUC-ROC | **0.8673** |
| Exactitud | 0.7780 |

Este resultado ubica al proyecto en el rango de **0.85 ≤ AUC-ROC < 0.87 → 5 SP** según los criterios de evaluación.

### Variables más relevantes para predecir churn

1. Antigüedad del cliente (meses) — 44.5%
2. Cargos mensuales (`MonthlyCharges`) — 12.3%
3. Cargos totales (`TotalCharges`) — 12.1%
4. Tipo de contrato de 2 años — 6.0%
5. Servicio de fibra óptica — 4.2%

## 💡 Recomendaciones de negocio

- Priorizar campañas de retención en los **primeros meses de contrato**, ventana donde se concentra el mayor riesgo de cancelación.
- Incentivar la **migración de contratos mes a mes hacia contratos anuales**.
- Revisar la estructura de precios para clientes con **cargos mensuales altos**.
- Investigar la satisfacción del segmento de **fibra óptica**.
- Usar las probabilidades del modelo como **sistema de alerta temprana** dentro del CRM de marketing.

## 🛠️ Tecnologías utilizadas

- Python (pandas, numpy)
- scikit-learn (preprocesamiento, validación cruzada, `GridSearchCV`, métricas)
- CatBoost, LightGBM
- matplotlib, seaborn

## 📁 Estructura del repositorio

```
├── Codigo_de_solucion_proyecto_final.ipynb   # Notebook con el desarrollo completo del modelo
├── README.md                                  # Este archivo
```

## 🚀 Cómo ejecutar

```bash
pip install pandas numpy scikit-learn catboost lightgbm matplotlib seaborn
jupyter notebook Codigo_de_solucion_proyecto_final.ipynb
```

## ✍️ Autor

**Juan Buri Loarte** — Jr. Geology and Mining Engineer con especialización en Ciencia de Datos.
[GitHub](https://github.com/loartejuan1994-jpg)

---

