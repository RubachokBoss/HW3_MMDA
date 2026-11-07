import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_model():
    return joblib.load(ROOT / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(ROOT / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


st.set_page_config(page_title='Подбор вина', layout='wide')
model = load_model()
test = load_test()
X = test.drop(columns='target')
y = test['target']
p = test_proba(model, X)
default = json.loads((ROOT / 'config.json').read_text())['threshold']
threshold = st.sidebar.slider('Порог отбора', 0.0, 1.0, default, step=0.001, format='%.4f')
st.title('Подбор вина')
single, quality, batch, factors = st.tabs(['Одно вино', 'Модельные метрики', 'Загрузка CSV', 'Коэффициенты'])

with single:
    values = {}
    columns = st.columns(3)
    for i, feature in enumerate(X.columns):
        column = columns[i % 3]
        if feature == 'is_red':
            values[feature] = int(column.radio('Цвет вина', ['Белое', 'Красное']) == 'Красное')
        else:
            series = X[feature]
            values[feature] = column.number_input(feature, float(series.min()), float(series.max()),
                                                   float(series.median()), step=0.001, format='%.5f')
    probability = model.predict_proba(pd.DataFrame([values]))[0, 1]
    st.metric('Вероятность хорошего вина', f'{probability:.1%}')
    st.write('Включить в подборку' if probability >= threshold else 'Пропустить')

with quality:
    prediction = p >= threshold
    precision = metrics.precision_score(y, prediction, zero_division=0)
    recall = metrics.recall_score(y, prediction)
    for col, label, value in zip(st.columns(3), ['Precision', 'Recall', 'F1'],
                                 [precision, recall, metrics.f1_score(y, prediction)]):
        if label == 'Precision' and not prediction.any():
            col.metric(label, 'Нет рекомендаций')
        else:
            col.metric(label, f'{value:.3f}')
    cm = metrics.confusion_matrix(y, prediction, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics.ConfusionMatrixDisplay(cm).plot(ax=axes[0], colorbar=False)
    metrics.RocCurveDisplay.from_predictions(y, p, ax=axes[1])
    axes[1].scatter(fp / (fp + tn), recall, color='red', zorder=3)
    metrics.PrecisionRecallDisplay.from_predictions(y, p, ax=axes[2])
    pr_marker = precision if prediction.any() else 1.0
    axes[2].scatter(recall, pr_marker, color='red', zorder=3)
    if not prediction.any():
        st.caption('При таком пороге подборка пуста. Precision не определён. На PR-кривой показана её крайняя точка (0, 1).')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption('Выше порог: отбор строже, подборка меньше. Ниже порог: находим больше хороших вин, но можем добавить больше посредственных.')

with batch:
    file = st.file_uploader('CSV с признаками вин, разделитель: запятая', type='csv')
    if file is not None:
        try:
            data = pd.read_csv(file)
            missing = X.columns.difference(data.columns)
            if len(missing):
                raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
            features = data[X.columns].apply(pd.to_numeric)
            if features.empty or not np.isfinite(features).all().all():
                raise ValueError('Нужна хотя бы одна строка с числами во всех признаках.')
            if not features['is_red'].isin([0, 1]).all():
                raise ValueError('В is_red допустимы только 0 и 1.')
            data['probability'] = model.predict_proba(features)[:, 1]
            data['selected'] = (data['probability'] >= threshold).astype(int)
            st.dataframe(data)
            st.download_button('Скачать результат', data.to_csv(index=False), 'predictions.csv', 'text/csv')
        except (ValueError, UnicodeError) as error:
            st.error(f'Проверьте CSV. {error}')

with factors:
    coef = pd.Series(model[-1].coef_[0], index=X.columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    coef.plot.barh(ax=ax, color=np.where(coef > 0, 'seagreen', 'indianred'))
    ax.axvline(0, color='black', linewidth=0.7)
    ax.set_xlabel('Коэффициент после масштабирования')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write(f'При прочих равных рост {coef.idxmax()} сильнее всего повышает оценку, '
             f'а рост {coef.idxmin()} сильнее всего её снижает.')

