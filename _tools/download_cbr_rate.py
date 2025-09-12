# _tools/download_cbr_rate.py

import pandas as pd
import requests
from datetime import datetime
import os
import argparse

# Импортируем 'requests' и 'pandas' для работы с HTTP-запросами и данными.
# 'datetime' и 'os' для работы с датами и путями к файлам.
# 'argparse' для обработки аргументов командной строки.

# В реальном проекте 'project_setup' может содержать настройки,
# но здесь мы определяем все необходимые импорты напрямую.

def download_and_save_cbr_key_rate(data_dir="1_data"):
    """
    Скачивает ПОЛНУЮ историю ключевой ставки с сайта ЦБ РФ,
    отправляя POST-запрос для получения всего диапазона дат.
    """
    URL = "https://www.cbr.ru/hd_base/keyrate/"
    print(f"Загрузка полной истории ключевой ставки со страницы: {URL}...")

    try:
        # Формируем POST-запрос с нужным диапазоном дат.
        # Начальная дата "01.01.1992" позволяет получить всю доступную историю.
        start_date_str = "01.01.1992"
        end_date_str = datetime.now().strftime('%d.%m.%Y')
        payload = {
            "UniDbQuery.Posted": "True",
            "UniDbQuery.From": start_date_str,
            "UniDbQuery.To": end_date_str
        }
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.post(URL, data=payload, headers=headers)
        response.raise_for_status()

        # Чтение HTML-таблицы.
        tables = pd.read_html(response.content, decimal=',')
        if not tables:
            print("❌ На странице не найдено таблиц с данными.")
            return

        df = tables[0]
        df.columns = ['datetime', 'key_rate']
        
        # Преобразуем 'datetime' в формат даты.
        df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y')
        
        # Преобразуем столбец 'key_rate' в числовой тип.
        # 'errors="coerce"' преобразует все нечисловые значения (включая
        # пустые строки, тире и т.д.) в NaN. Это критически важный шаг
        # для правильного заполнения пропусков.
        df['key_rate'] = pd.to_numeric(df['key_rate'], errors='coerce')

        # Создаем полный календарь и сливаем с ним полученные данные.
        start_date = df['datetime'].min()
        end_date = datetime.now()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        full_df = pd.DataFrame(full_date_range, columns=['datetime'])
        
        # Объединяем полные даты с данными по ключевой ставке.
        merged_df = pd.merge(full_df, df, on='datetime', how='left')
        
        # Заполняем пропуски предыдущими значениями (forward fill).
        # Теперь, когда пропуски стали NaN, этот метод сработает корректно.
        merged_df['key_rate'].ffill(inplace=True)
        
        output_path = os.path.join(data_dir, "CBR_KEY_RATE.csv")
        merged_df[['datetime', 'key_rate']].to_csv(output_path, index=False)
        print(f"✅ Полная история ключевой ставки успешно сохранена в {output_path}")

    except requests.exceptions.RequestException as http_err:
        print(f"❌ Ошибка HTTP при доступе к странице: {http_err}")
    except Exception as e:
        print(f"❌ Не удалось загрузить данные по ключевой ставке. Ошибка: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Загрузка ПОЛНОЙ истории ключевой ставки ЦБ РФ.")
    parser.add_argument('--data_dir', type=str, default="1_data", help='Папка для сохранения данных.')
    args = parser.parse_args()
    
    os.makedirs(args.data_dir, exist_ok=True)
    download_and_save_cbr_key_rate(args.data_dir)