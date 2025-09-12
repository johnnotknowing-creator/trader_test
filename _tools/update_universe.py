from _core.paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, TOOLS_DIR, CONFIGS_DIR, ensure_dirs, load_dotenv_if_exists
from _core.libs import *
load_dotenv_if_exists()
ensure_dirs()
# _tools/update_universe.py

def get_moex_tickers():
    """
    Получает список всех торгуемых акций с Московской биржи, 
    с динамическим определением столбцов и логированием проблемных ответов.
    """
    print("Запрашиваю актуальный список акций с Московской биржи...")
    
    url = "https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities.json"
    params = {
        "iss.meta": "off",
        "iss.only": "securities"
    }
    
    all_tickers = []
    start = 0
    last_page_content = None

    while True:
        data = {}
        try:
            page_params = params.copy()
            page_params['start'] = start
            
            response = requests.get(url, params=page_params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if 'securities' not in data or not data['securities']['columns']:
                print("❌ API не вернул ожидаемый блок 'securities'.")
                return None
            
            columns = data['securities']['columns']
            securities_data = data['securities']['data']
            
            try:
                secid_index = columns.index('SECID')
                # ⏬ --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ --- ⏬
                type_index = columns.index('SECTYPE') # Ищем 'SECTYPE' вместо 'TYPE'
                # ⏫ --- КОНЕЦ ИСПРАВЛЕНИЯ --- ⏫
            except ValueError:
                print("❌ В ответе API отсутствуют необходимые столбцы 'SECID' или 'SECTYPE'.")
                
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                logs_dir = os.path.join(project_root, "_logs")
                os.makedirs(logs_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                log_file = os.path.join(logs_dir, f"api_response_{timestamp}.json")
                
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                
                print(f"⚠️  Полный ответ от API сохранен для анализа в файл: {log_file}")
                return None

            current_page_content = json.dumps(securities_data)
            if not securities_data or current_page_content == last_page_content:
                break
            last_page_content = current_page_content
            
            for security in securities_data:
                if len(security) > max(secid_index, type_index):
                    ticker = security[secid_index]
                    stock_type = security[type_index]
                    
                    # '1'-Common, '2'-Preferred, 'D'-Depositary Receipt. MOEX также использует 'J' для акций.
                    if stock_type in ['1', '2', 'D', 'J']:
                        all_tickers.append(ticker)
            
            start += len(securities_data)
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Сетевая ошибка при запросе данных с Мосбиржи: {e}")
            return None
        except Exception as e:
            print(f"❌ Непредвиденная ошибка: {e}")
            return None
            
    if not all_tickers:
        print("⚠️  Не удалось найти ни одного активного тикера. Возможно, временные проблемы с API Мосбиржи.")
    else:
        print(f"✅ Получено {len(all_tickers)} активных тикеров.")

    return sorted(list(set(all_tickers)))

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    universe_file_path = os.path.join(project_root, "universe.csv")
    
    tickers = get_moex_tickers()
    
    if tickers is not None:
        # Проверяем, что список не пустой перед сохранением
        if tickers:
            universe_df = pd.DataFrame(tickers, columns=['ticker'])
            universe_df.to_csv(universe_file_path, index=False)
            print(f"✅ Список тикеров успешно сохранен в файл: {universe_file_path}")
        else:
            print("⚠️ Итоговый список тикеров пуст. Файл universe.csv не был изменен.")
