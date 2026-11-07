# Подбор вина

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r wine_app/requirements.txt
.\.venv\Scripts\python -m streamlit run wine_app/app.py
```


Модель хранится в cache_resource, чтобы все сеансы использовали один загруженный объект. Без декоратора joblib.load читал бы файл при каждом движении слайдера. Для таблицы и вероятностей используем cache_data: Streamlit сохраняет результат вычисления и возвращает копию.

Данные: [Wine Quality, P. Cortez et al. (2009), CC BY 4.0](https://archive.ics.uci.edu/dataset/186/wine+quality). Целевой класс 1 соответствует оценке от 7.

