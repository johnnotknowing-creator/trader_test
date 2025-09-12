from _core.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, TOOLS_DIR, CONFIGS_DIR, ensure_dirs, load_dotenv_if_exists
from _core.libs import *
load_dotenv_if_exists()
ensure_dirs()
# _tools/download_all_data.py
# Явные импорты из модулей того же пакета (_tools)
from _tools.universal_downloader import download_all_stocks
from _tools.download_index import download_moex_index_history
from _tools.download_cbr_rate import download_and_save_cbr_key_rate

CONFIG = {
    "YEARS": 20,
    "DATA_DIR": "1_data",
    "UNIVERSE_FILE": "universe.csv"
}

if __name__ == "__main__":
    print(f"--- Начинаю полную загрузку данных за {CONFIG['YEARS']} лет ---")
    
    os.makedirs(CONFIG["DATA_DIR"], exist_ok=True)
    
    print("\n[1/3] Загрузка котировок по акциям...")
    # Загружаем список тикеров из файла
    universe_path = os.path.join(PROJECT_ROOT, CONFIG["UNIVERSE_FILE"])
    try:
        tickers = pd.read_csv(universe_path)['ticker'].tolist()
    except FileNotFoundError:
        print(f"❌ Файл со списком акций не найден: {universe_path}. Запустите сначала update_universe.py")
        exit()

    # Передаем только список тикеров в функцию загрузки
    download_all_stocks(tickers)
    
    print("\n[2/3] Загрузка истории индекса IMOEX...")
    download_moex_index_history(CONFIG["YEARS"], CONFIG["DATA_DIR"])
    
    print("\n[3/3] Загрузка истории ключевой ставки ЦБ РФ...")
    download_and_save_cbr_key_rate(CONFIG["DATA_DIR"])
    
    print("\n✅ Все данные успешно загружены и обновлены.")
